import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteStaticCopy } from "vite-plugin-static-copy";

export default defineConfig({
	base: "./",
	plugins: [
		react(),
		viteStaticCopy({
			targets: [
				{
					src: "pkg/**/*",
					dest: "pkg",
				},
				{
					// Ensure wasm sits next to bundled JS chunks under assets/
					src: "pkg/webapp_bg.wasm",
					dest: "assets",
				},
				{
					// Extra copy at dist root for any relative fallbacks
					src: "pkg/webapp_bg.wasm",
					dest: ".",
				},
			],
		}),
	],
	build: {
		outDir: "dist",
		assetsDir: "assets",
		sourcemap: true,
	},
});
