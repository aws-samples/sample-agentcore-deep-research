// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],

  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },

  build: {
    outDir: "build",
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (
            id.includes("node_modules/react-dom") ||
            id.includes("node_modules/react/") ||
            id.includes("node_modules/react-router-dom")
          ) {
            return "react-vendor";
          }
          if (
            id.includes("node_modules/@radix-ui/react-dialog") ||
            id.includes("node_modules/@radix-ui/react-select") ||
            id.includes("node_modules/@radix-ui/react-alert-dialog") ||
            id.includes("node_modules/@radix-ui/react-progress")
          ) {
            return "ui-vendor";
          }
          if (
            id.includes("node_modules/react-oidc-context") ||
            id.includes("node_modules/aws-amplify")
          ) {
            return "auth-vendor";
          }
        },
      },
    },
  },

  server: {
    port: 3000,
    open: !process.env.DOCKER_CONTAINER,
  },
});
