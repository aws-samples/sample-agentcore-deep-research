#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.


"""
Cross-platform frontend deployment script.

Deploys the React frontend to AWS Amplify by:
1. Fetching configuration from CDK stack outputs
2. Generating aws-exports.json
3. Building the frontend
4. Packaging and uploading to S3
5. Triggering Amplify deployment

Requires: Python 3.11+, AWS CLI, npm, Node.js
No external Python dependencies - uses standard library only.
"""

import atexit
import json
import os
import re
import shutil
import subprocess  # nosec B404 - subprocess used securely with explicit parameters
import sys
import time
from pathlib import Path

# Minimum Python version check
if sys.version_info < (3, 8):
    print("Error: Python 3.8 or higher is required")
    sys.exit(1)

# Constants
BRANCH_NAME = "main"
NEXT_BUILD_DIR = "build"
CLEANUP_FILES: list = []


# --- Logging helpers ---


def log_info(message: str) -> None:
    """Print an info message."""
    print(f"ℹ {message}")


def log_success(message: str) -> None:
    """Print a success message."""
    print(f"✓ {message}")


def log_error(message: str) -> None:
    """Print an error message to stderr."""
    print(f"✗ {message}", file=sys.stderr)


def log_warning(message: str) -> None:
    """Print a warning message."""
    print(f"⚠ {message}")


# --- Utility functions ---


def cleanup() -> None:
    """Remove temporary files created during deployment."""
    for filepath in CLEANUP_FILES:
        if os.path.exists(filepath):
            os.remove(filepath)
            log_info(f"Cleaned up {filepath}")


def run_command(
    command: list,
    capture_output: bool = True,
    check: bool = True,
    cwd: str | None = None,
) -> subprocess.CompletedProcess:
    """
    Execute a command securely via subprocess.

    Args:
        command: List of command arguments
        capture_output: Whether to capture stdout/stderr
        check: Whether to raise on non-zero exit
        cwd: Working directory for the command

    Returns:
        CompletedProcess instance with command results
    """
    return subprocess.run(  # nosec B603 - command constructed from safe list
        command,
        capture_output=capture_output,
        text=True,
        check=check,
        shell=False,
        timeout=300,
        cwd=cwd,
    )


def check_prerequisite(command: str) -> bool:
    """
    Check if a command is available in PATH.

    Args:
        command: Name of the command to check

    Returns:
        True if command exists, False otherwise
    """
    return shutil.which(command) is not None


def parse_config_yaml(config_path: Path) -> dict:
    """
    Parse config.yaml using regex (no PyYAML dependency).

    Args:
        config_path: Path to config.yaml file

    Returns:
        Dictionary with stack_name_base, pattern, and tools values
    """
    config: dict = {
        "stack_name_base": "",
        "region": None,
        "pattern": "strands-deep-research",
        "auto_deploy_frontend": False,
        "tools": {},
    }

    if not config_path.exists():
        return config

    content = config_path.read_text()

    # Extract stack_name_base
    match = re.search(r"^stack_name_base:\s*(\S+)", content, re.MULTILINE)
    if match:
        config["stack_name_base"] = match.group(1).strip("\"'")

    # Extract region
    match = re.search(r"^region:\s*(\S+)", content, re.MULTILINE)
    if match:
        val = match.group(1).split("#")[0].strip().strip("\"'")
        config["region"] = val if val != "null" else None

    # Extract pattern from backend section
    match = re.search(r"pattern:\s*(\S+)", content)
    if match:
        config["pattern"] = match.group(1).split("#")[0].strip().strip("\"'")

    # Extract auto_deploy_frontend flag
    match = re.search(r"^auto_deploy_frontend:\s*(true|false)", content, re.MULTILINE)
    if match:
        config["auto_deploy_frontend"] = match.group(1) == "true"

    # Extract tools config (tool_name -> {enabled, default_on, required: {...}})
    tools_section = re.search(r"^tools:\s*\n((?:[ \t]+\S.*\n)*)", content, re.MULTILINE)
    if tools_section:
        tools_text = tools_section.group(1)
        current_tool = None
        in_required = False
        for line in tools_text.split("\n"):
            # Match tool name (2-space indented)
            tool_match = re.match(r"^  (\w+):\s*$", line)
            if tool_match:
                current_tool = tool_match.group(1)
                in_required = False
                config["tools"][current_tool] = {
                    "enabled": True,
                    "default_on": False,
                }
                continue
            if not current_tool:
                continue
            # Match "required:" block start (4-space indented)
            if re.match(r"^    required:\s*$", line):
                in_required = True
                config["tools"][current_tool]["required"] = {}
                continue
            if in_required:
                # Match required fields (6-space indented)
                req_match = re.match(r"^      (\w+):\s*(\S+)", line)
                if req_match:
                    val = req_match.group(2).split("#")[0].strip().strip("\"'")
                    config["tools"][current_tool]["required"][req_match.group(1)] = (
                        val if val != "null" else None
                    )
                    continue
                in_required = False
            # Match enabled/default_on booleans (4-space indented)
            prop_match = re.match(r"^    (enabled|default_on):\s*(true|false)", line)
            if prop_match:
                config["tools"][current_tool][prop_match.group(1)] = (
                    prop_match.group(2) == "true"
                )

    return config


def get_file_size_human(filepath: str) -> str:
    """
    Get human-readable file size.

    Args:
        filepath: Path to the file

    Returns:
        Human-readable size string (e.g., "1.5MB")
    """
    size: float = os.path.getsize(filepath)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


# --- AWS CLI wrappers ---


def get_stack_outputs(stack_name: str, region: str | None = None) -> tuple[dict[str, str], str]:
    """
    Fetch CloudFormation stack outputs via AWS CLI.

    Args:
        stack_name: Name of the CloudFormation stack
        region: AWS region (optional, uses CLI default if None)

    Returns:
        Tuple of (outputs dict, region string)
    """
    cmd = [
        "aws",
        "cloudformation",
        "describe-stacks",
        "--stack-name",
        stack_name,
        "--output",
        "json",
    ]
    if region:
        cmd.extend(["--region", region])
    result = run_command(cmd)

    stack_data = json.loads(result.stdout)
    stacks = stack_data.get("Stacks", [])
    if not stacks:
        raise ValueError(f"Stack '{stack_name}' not found or has no data")
    outputs = stacks[0].get("Outputs", [])

    # Extract region from stack ARN if not provided
    if not region:
        stack_arn = stacks[0]["StackId"]
        arn_parts = stack_arn.split(":")
        if len(arn_parts) < 4:
            raise ValueError(f"Invalid stack ARN format: {stack_arn}")
        region = arn_parts[3]

    return {o["OutputKey"]: o["OutputValue"] for o in outputs}, region


def upload_to_s3(local_path: str, bucket: str, key: str, region: str | None = None) -> None:
    """
    Upload a file to S3 via AWS CLI.

    Args:
        local_path: Path to local file
        bucket: S3 bucket name
        key: S3 object key
        region: AWS region (optional)
    """
    cmd = ["aws", "s3", "cp", local_path, f"s3://{bucket}/{key}", "--no-progress"]
    if region:
        cmd.extend(["--region", region])
    run_command(cmd)


def start_amplify_deployment(app_id: str, branch: str, source_url: str, region: str | None = None) -> dict:
    """
    Start an Amplify deployment via AWS CLI.

    Args:
        app_id: Amplify application ID
        branch: Branch name to deploy
        source_url: S3 URL of deployment package
        region: AWS region (optional)

    Returns:
        Deployment response as dictionary
    """
    cmd = [
        "aws", "amplify", "start-deployment",
        "--app-id", app_id,
        "--branch-name", branch,
        "--source-url", source_url,
        "--output", "json",
    ]
    if region:
        cmd.extend(["--region", region])
    result = run_command(cmd)

    return dict(json.loads(result.stdout))


def get_amplify_job_status(app_id: str, branch: str, job_id: str, region: str | None = None) -> str:
    """
    Get the status of an Amplify deployment job.

    Args:
        app_id: Amplify application ID
        branch: Branch name
        job_id: Deployment job ID
        region: AWS region (optional)

    Returns:
        Job status string
    """
    cmd = [
        "aws", "amplify", "get-job",
        "--app-id", app_id,
        "--branch-name", branch,
        "--job-id", job_id,
        "--output", "json",
    ]
    if region:
        cmd.extend(["--region", region])
    result = run_command(cmd)

    return str(json.loads(result.stdout)["job"]["summary"]["status"])


def get_amplify_app_domain(app_id: str, region: str | None = None) -> str:
    """
    Get the default domain for an Amplify app.

    Args:
        app_id: Amplify application ID
        region: AWS region (optional)

    Returns:
        Default domain string
    """
    cmd = [
        "aws", "amplify", "get-app",
        "--app-id", app_id,
        "--query", "app.defaultDomain",
        "--output", "text",
    ]
    if region:
        cmd.extend(["--region", region])
    result = run_command(cmd)

    return str(result.stdout.strip())


# --- Main deployment logic ---


def generate_aws_exports(
    stack_name: str,
    outputs: dict[str, str],
    region: str,
    pattern: str,
    frontend_dir: Path,
    tools: dict | None = None,
) -> None:
    """
    Generate aws-exports.json configuration file.

    Args:
        stack_name: CloudFormation stack name
        outputs: Stack outputs dictionary
        region: AWS region
        pattern: Agent pattern name
        frontend_dir: Path to frontend directory
        tools: Tools configuration from config.yaml
    """
    required = [
        "CognitoClientId",
        "CognitoUserPoolId",
        "AmplifyUrl",
        "RuntimeArn",
        "FeedbackApiUrl",
    ]
    missing = [k for k in required if k not in outputs]

    if missing:
        raise ValueError(f"Missing required stack outputs: {', '.join(missing)}")

    aws_exports: dict = {
        "authority": f"https://cognito-idp.{region}.amazonaws.com/{outputs['CognitoUserPoolId']}",
        "client_id": outputs["CognitoClientId"],
        "redirect_uri": outputs["AmplifyUrl"],
        "post_logout_redirect_uri": outputs["AmplifyUrl"],
        "response_type": "code",
        "scope": "email openid profile",
        "automaticSilentRenew": True,
        "agentRuntimeArn": outputs["RuntimeArn"],
        "awsRegion": region,
        "feedbackApiUrl": outputs["FeedbackApiUrl"],
        "agentPattern": pattern,
    }
    if tools:
        aws_exports["tools"] = tools

    public_dir = frontend_dir / "public"
    public_dir.mkdir(parents=True, exist_ok=True)

    output_path = public_dir / "aws-exports.json"
    output_path.write_text(json.dumps(aws_exports, indent=2))

    log_success(f"Generated aws-exports.json at {output_path}")


def create_deployment_zip(build_dir: Path, output_path: Path) -> None:
    """
    Create a zip archive of the build directory.

    Args:
        build_dir: Path to the build directory
        output_path: Path for the output zip file (without .zip extension)
    """
    # shutil.make_archive adds .zip automatically
    shutil.make_archive(
        str(output_path.with_suffix("")), "zip", root_dir=str(build_dir)
    )


def main() -> int:
    """
    Main deployment function.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    atexit.register(cleanup)

    # Determine paths
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    frontend_dir = project_root / "frontend"
    config_path = project_root / "infra-cdk" / "config.yaml"
    if not config_path.exists():
        example_path = project_root / "infra-cdk" / ".config_example.yaml"
        if example_path.exists():
            log_warning(
                "config.yaml not found, using .config_example.yaml defaults. "
                "To customize, run: cp infra-cdk/.config_example.yaml infra-cdk/config.yaml"
            )
            config_path = example_path
        else:
            log_error(
                "No configuration file found. Create config.yaml by copying the example:\n"
                "  cp infra-cdk/.config_example.yaml infra-cdk/config.yaml"
            )
            return 1

    log_info("🚀 Starting frontend deployment process...")
    print()

    # Validate prerequisites
    log_info("Validating prerequisites...")
    prerequisites = ["npm", "aws", "node"]
    for prereq in prerequisites:
        if not check_prerequisite(prereq):
            log_error(f"{prereq} is not installed")
            return 1
    log_success("All prerequisites found")

    # Verify AWS credentials are configured
    log_info("Verifying AWS credentials...")
    try:
        run_command(["aws", "sts", "get-caller-identity"], capture_output=True)
        log_success("AWS credentials configured")
    except subprocess.CalledProcessError:
        log_error("AWS credentials not configured or invalid")
        log_info("Run 'aws configure' to set up your AWS credentials")
        log_info(
            "Or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables"
        )
        return 1

    # Get stack name
    stack_name = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("STACK_NAME")

    if not stack_name:
        config = parse_config_yaml(config_path)
        stack_name = config.get("stack_name_base")

    if not stack_name:
        log_error("Stack name is required")
        log_info("Usage: python deploy-frontend.py <stack-name>")
        log_info("   or: STACK_NAME=your-stack ./deploy-frontend.py")
        return 1

    # Fetch CDK outputs
    log_info(f"Fetching configuration from CDK stack: {stack_name}")
    config = parse_config_yaml(config_path)
    config_region = config.get("region")
    try:
        outputs, region = get_stack_outputs(stack_name, config_region)
    except subprocess.CalledProcessError as e:
        log_error(f"Failed to fetch stack outputs: {e.stderr}")
        return 1
    except ValueError as e:
        log_error(str(e))
        return 1

    # Validate required outputs
    app_id = outputs.get("AmplifyAppId")
    deployment_bucket = outputs.get("StagingBucketName")

    if not app_id:
        log_error("Could not find Amplify App ID in stack outputs")
        return 1
    if not deployment_bucket:
        log_error("Could not find Staging Bucket Name in stack outputs")
        return 1

    log_success(f"App ID: {app_id}")
    log_success(f"Staging Bucket: {deployment_bucket}")
    log_success(f"Region: {region}")

    # Get agent pattern and tools config
    pattern = config.get("pattern", "strands-deep-research")
    tools = config.get("tools", {})
    log_info(f"Agent pattern: {pattern}")

    # Generate aws-exports.json
    log_info("Generating aws-exports.json...")
    try:
        generate_aws_exports(stack_name, outputs, region, pattern, frontend_dir, tools)
    except ValueError as e:
        log_error(str(e))
        return 1

    # Change to frontend directory
    os.chdir(frontend_dir)
    log_info(f"Working directory: {frontend_dir}")

    # Install dependencies if needed
    node_modules = frontend_dir / "node_modules"
    package_json = frontend_dir / "package.json"

    if (
        not node_modules.exists()
        or package_json.stat().st_mtime > node_modules.stat().st_mtime
    ):
        log_info("Installing dependencies...")
        try:
            run_command(["npm", "install"], capture_output=False)
            log_success("Dependencies installed")
        except subprocess.CalledProcessError:
            log_error("Failed to install dependencies")
            return 1
    else:
        log_success("Dependencies are up to date")

    # Build frontend
    log_info("Building React app...")
    try:
        run_command(["npm", "run", "build"], capture_output=False)
        log_success("Build completed")
    except subprocess.CalledProcessError:
        log_error("Build failed")
        return 1

    # Verify build directory
    build_dir = frontend_dir / NEXT_BUILD_DIR
    if not build_dir.exists():
        log_error(f"Build directory '{NEXT_BUILD_DIR}' not found")
        return 1

    # Copy aws-exports.json to build
    aws_exports_src = frontend_dir / "public" / "aws-exports.json"
    aws_exports_dst = build_dir / "aws-exports.json"
    shutil.copy2(aws_exports_src, aws_exports_dst)
    log_success("Added aws-exports.json to build directory")

    # Create deployment zip
    log_info("Creating deployment package...")
    zip_path = frontend_dir / "amplify-deploy.zip"
    CLEANUP_FILES.append(str(zip_path))

    create_deployment_zip(build_dir, zip_path)
    zip_size = get_file_size_human(str(zip_path))
    log_success(f"Package created ({zip_size})")

    # Upload to S3
    s3_key = f"amplify-deploy-{int(time.time())}.zip"
    log_info(f"Uploading to S3 (s3://{deployment_bucket}/{s3_key})...")
    try:
        upload_to_s3(str(zip_path), deployment_bucket, s3_key, region)
        log_success("Upload completed")
    except subprocess.CalledProcessError as e:
        log_error(f"S3 upload failed: {e.stderr}")
        return 1

    # Start Amplify deployment
    log_info("Starting Amplify deployment...")
    source_url = f"s3://{deployment_bucket}/{s3_key}"

    try:
        deployment = start_amplify_deployment(app_id, BRANCH_NAME, source_url, region)
        job_id = deployment["jobSummary"]["jobId"]
        log_success(f"Deployment initiated (Job ID: {job_id})")
    except subprocess.CalledProcessError as e:
        log_error(f"Amplify deployment failed: {e.stderr}")
        return 1

    # Poll deployment status
    log_info("Monitoring deployment status...")
    while True:
        try:
            status = get_amplify_job_status(app_id, BRANCH_NAME, job_id, region)
        except subprocess.CalledProcessError as e:
            log_error(f"Failed to get deployment status: {e.stderr}")
            return 1

        print(f"  Status: {status}")

        if status == "SUCCEED":
            log_success("Deployment completed successfully!")
            break
        elif status in ("FAILED", "CANCELLED"):
            log_error(f"Deployment {status.lower()}")
            return 1

        time.sleep(10)

    # Print final info
    print()
    log_info(f"S3 Package: s3://{deployment_bucket}/{s3_key}")
    log_info("Console: https://console.aws.amazon.com/amplify/apps")
    try:
        app_domain = get_amplify_app_domain(app_id, region)
        log_info(f"App URL: https://{BRANCH_NAME}.{app_domain}")
    except subprocess.CalledProcessError:
        log_warning("Could not retrieve app URL - check Amplify console")

    return 0


if __name__ == "__main__":
    sys.exit(main())
