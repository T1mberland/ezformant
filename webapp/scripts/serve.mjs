import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(new URL(import.meta.url)));
const root = path.resolve(__dirname, "..", "dist");
const port = Number(process.env.PORT ?? 5173);

const mimeTypes = {
	".html": "text/html",
	".js": "text/javascript",
	".mjs": "text/javascript",
	".css": "text/css",
	".wasm": "application/wasm",
	".json": "application/json",
};

const server = createServer((req, res) => {
	if (!req.url) {
		res.writeHead(400);
		res.end("Bad request");
		return;
	}

	const url = new URL(req.url, "http://localhost");
	let pathname = decodeURIComponent(url.pathname);
	if (pathname === "/") {
		pathname = "index.html";
	} else if (pathname.startsWith("/")) {
		pathname = pathname.slice(1);
	}

	const filePath = path.resolve(root, pathname);
	if (!filePath.startsWith(root)) {
		res.writeHead(403);
		res.end("Forbidden");
		return;
	}

	if (!existsSync(filePath) || !statSync(filePath).isFile()) {
		res.writeHead(404);
		res.end("Not found");
		return;
	}

	const ext = path.extname(filePath);
	const contentType = mimeTypes[ext] ?? "application/octet-stream";
	res.writeHead(200, { "Content-Type": contentType });
	createReadStream(filePath).pipe(res);
});

server.listen(port, () => {
	console.log(`Serving dist/ at http://localhost:${port}`);
});
