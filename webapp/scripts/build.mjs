import { execSync } from "node:child_process";
import { cpSync, existsSync, mkdirSync, rmSync } from "node:fs";
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

console.log("Build complete. Output available in dist/");
