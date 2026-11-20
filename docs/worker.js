// This file runs off the main page, in a Web Worker.
// Pyodide and the Monte Carlo search live here so the board stays clickable
// while the agent thinks. There is no Python server: this worker is the runtime.

const PYODIDE_VERSION = "0.29.0";
const INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

importScripts(`${INDEX_URL}pyodide.js`);

let pyodide = null;

self.onmessage = async (event) => {
  const message = event.data;

  if (message.type === "init") {
    try {
      pyodide = await loadPyodide({ indexURL: INDEX_URL });
      writeGameFiles(message.files);
      pyodide.runPython(`
import sys
sys.path.insert(0, "/game")
import web_game
`);
      self.postMessage({ type: "ready" });
    } catch (error) {
      self.postMessage({ type: "init-error", error: String(error) });
    }
    return;
  }

  if (message.type === "call") {
    try {
      // A JS object is not a Python dict. Send JSON text and let Python parse it.
      pyodide.globals.set("request_name", message.name);
      pyodide.globals.set("request_payload", JSON.stringify(message.payload ?? {}));
      const result = pyodide.runPython("web_game.handle(request_name, request_payload)");
      self.postMessage({ type: "result", id: message.id, ok: true, result: String(result) });
    } catch (error) {
      self.postMessage({
        type: "result",
        id: message.id,
        ok: false,
        error: String(error),
      });
    }
  }
};

function writeGameFiles(files) {
  for (const [path, content] of Object.entries(files)) {
    const directory = path.slice(0, path.lastIndexOf("/"));
    if (directory) {
      pyodide.FS.mkdirTree(directory);
    }
    pyodide.FS.writeFile(path, content);
  }
}
