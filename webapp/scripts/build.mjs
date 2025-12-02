import { execSync } from "node:child_process";
import {
	cpSync,
	existsSync,
	mkdirSync,
	readFileSync,
	rmSync,
	writeFileSync,
} from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(new URL(import.meta.url)));
const projectRoot = path.resolve(__dirname, "..");
const distDir = path.join(projectRoot, "dist");

if (existsSync(distDir)) {
	rmSync(distDir, { recursive: true, force: true });
}

execSync("npx tsc --project tsconfig.json", {
	cwd: projectRoot,
	stdio: "inherit",
});
execSync("npx tsc --project tsconfig.worker.json", {
	cwd: projectRoot,
	stdio: "inherit",
});

mkdirSync(distDir, { recursive: true });
cpSync(path.join(projectRoot, "index.html"), path.join(distDir, "index.html"));
cpSync(path.join(projectRoot, "pkg"), path.join(distDir, "pkg"), {
	recursive: true,
});

// Ensure compiled JS points to pkg inside dist (wasm-pack output is copied above)
for (const file of ["main.js", "formantWorker.js"]) {
	const filePath = path.join(distDir, file);
	if (!existsSync(filePath)) continue;
	const content = readFileSync(filePath, "utf8");
	const updated = content.replace(/\.\.\/pkg\//g, "./pkg/");
	writeFileSync(filePath, updated, "utf8");
}

console.log("Build complete. Output available in dist/");
