import Server.Common
import Lean.Data.Json

import Server.Rules

open Lean
open Lean.Json

def next_state (s : State) : Grid :=
  { data :=
      -- Build a GRID_SIZE × GRID_SIZE array of new cell‐values
      Array.ofFn (fun (r : Fin GRID_SIZE) =>
        Array.ofFn (fun (c : Fin GRID_SIZE) =>
          let pos : Coords := ⟨r.val, c.val⟩
          if empty s pos then
            EMPTY
          else if tree s pos then
            TREE
          else if fire1 s pos then
            FIRE_1
          else if fire2 s pos then
            FIRE_2
          else if fire3 s pos then
            FIRE_3
          else
            s.grid.get pos ) ) }

def next_states_parallel (states : Array State) : Array Grid :=
  let tasks : Array (Task Grid) :=
    states.map fun s =>
      Task.spawn fun _ =>
        next_state s
  tasks.map fun t => t.get


def printFlush [ToString α] (s : α) : IO Unit := do
  let out ← IO.getStdout
  out.putStr (toString s)
  out.putStr "\n\n"
  out.flush

/--
  Given a single line of input (a JSON string), attempt to parse it as an Array State,
  compute the next grids, and emit either an error object or a success object.
-/
def processLine (line : String) : IO Unit := do
  match Json.parse line with
  | Except.error errMsg =>
    let errJson := Json.mkObj [("error", Json.str s!"JSON parse error: {errMsg}")]
    printFlush (errJson.pretty)
  | Except.ok json =>
    match fromJson? (α := Array State) json with
    | Except.error errMsg2 =>
      let errJson := Json.mkObj [("error", Json.str s!"FromJson error: {errMsg2}")]
      printFlush (errJson.pretty)
    | Except.ok states =>
      let nextGrids : Array Grid := next_states_parallel states
      let outJson := Json.mkObj [("next_grids", toJson nextGrids)]
      printFlush (outJson.pretty)

/--
  A REPL loop: read lines from stdin until EOF, for each line run `processLine`.
  Uses `IO.getStdin` + `getLine?` to detect EOF.
-/
partial def loop (stdin : IO.FS.Stream) : IO Unit := do
  let line ← stdin.getLine
  if line = "" then
    return
  processLine line
  loop stdin

/--
  The program’s `main` just grabs stdin and starts the REPL.
-/
def main : IO Unit := do
  let stdinHandle ← IO.getStdin
  loop stdinHandle
