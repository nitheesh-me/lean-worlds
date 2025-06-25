import Lean
import Lean.Parser
import Lean.Elab.Command
import Lean.Data.Json
import Lean.Data.Json.Basic
import Lean.Data.RBTree
import Std
import Lean.Parser
import Lean.Meta
import Lean.Exception

import Server.Common

open Lean
open Lean.Json

namespace Server


def squareUci (uci : String) : Coords :=
  let uciArr := uci.toList
  let file := (uciArr[0]!.toNat - 'a'.toNat)
  let rank := (uciArr[1]!.toNat - '1'.toNat)
  { file := file, rank := rank }

def moveUci (uci : String) : Coords × Coords :=
  let uciArr := uci.toList
  let a := squareUci (uciArr[0]!.toString ++ uciArr[1]!.toString)
  let b := squareUci (uciArr[2]!.toString ++ uciArr[3]!.toString)
  (a, b)


def exampleBoard : Board := {
  data := (#[
    #[ (.rook, .black), (.knight, .black), (.bishop, .black), (.queen, .black), (.king,  .black), (.bishop, .black), (.knight, .black), (.rook,  .black) ],
    #[(.pawn, .black), (.pawn, .black), (.pawn, .black), (.pawn, .black), (.pawn, .black), (.pawn, .black), (.pawn, .black), (.pawn, .black)],
    #[ (.empty, .empty), (.empty, .empty), (.empty, .empty), (.empty, .empty), (.empty, .empty), (.empty, .empty), (.empty, .empty), (.empty, .empty) ],
    #[ (.empty, .empty), (.empty, .empty), (.empty, .empty), (.empty, .empty), (.empty, .empty), (.empty, .empty), (.empty, .empty), (.empty, .empty) ],
    #[ (.empty, .empty), (.empty, .empty), (.empty, .empty), (.empty, .empty), (.empty, .empty), (.empty, .empty), (.empty, .empty), (.empty, .empty) ],
    #[ (.empty, .empty), (.empty, .empty), (.empty, .empty), (.empty, .empty), (.empty, .empty), (.empty, .empty), (.empty, .empty), (.empty, .empty) ],
    #[(.pawn, .white), (.pawn, .white), (.pawn, .white), (.pawn, .white), (.pawn, .white), (.pawn, .white), (.pawn, .white), (.pawn, .white)],
    #[ (.rook,  .white), (.knight, .white), (.bishop, .white), (.queen, .white), (.king,  .white), (.bishop, .white), (.knight, .white), (.rook,  .white) ]
  ]).reverse  -- reverse the board so that we can index from bottom left
}

-- Experience structure for loading data
structure Experience where
  board : Board
  move : String
  legal : Bool
deriving FromJson, ToJson

-- Input command structure
structure EvaluationInput where
  premises : List String
deriving FromJson, ToJson

-- Evaluation result structure
structure EvaluationResult where
  premise : String
  true_count : Nat
deriving ToJson

-- Function to load experiences from a .jsonl file
def loadExperiences (filename : System.FilePath) : IO (List Experience) := do
  let lines ← IO.FS.lines filename
  let mut experiences : List Experience := []
  for line in lines do
    match Json.parse line >>= fromJson? with
    | Except.ok exp => experiences := exp :: experiences
    | Except.error err => IO.eprintln s!"Error parsing experience: {err} in line: {line}"
  return experiences.reverse -- Maintain original order


abbrev Rule := Board → Coords → Coords → Bool

def sampleRule : Rule := fun board a b => true
def f : Rule := (fun (board : Board) (a b : Coords) => board.piece a = .rook ∧ a.file = b.file)
abbrev g (board : Board) (a b : Coords) := board.piece a = .rook ∧ a.file = b.file

unsafe def parsePremise (s : String) : Lean.Elab.TermElabM Rule := do
  let sLam := s!"fun (board : Board) (a : Coords) (b : Coords) => ({s})"
  IO.println s!"{sLam}"

  -- -- Parsing
  let env: Lean.Environment ← Lean.getEnv
  let stx?: Except String Lean.Syntax := Lean.Parser.runParserCategory env `term sLam
  let stx : Lean.Syntax ← Lean.ofExcept stx?
  IO.println s!"{stx}"

  -- Elaboration
  let e ← Lean.Elab.Term.elabTerm stx .none
  IO.println s!"{e}"

  -- TODO: maybe this is redundant
  let e ← Lean.instantiateMVars e
  IO.println s!"{e}"

  -- Evaluation
  Lean.Meta.evalExprCore Rule e (fun _ => pure ()) (safety := Lean.DefinitionSafety.unsafe)

unsafe def parsePremiseLifted (s : String) : Lean.Elab.Command.CommandElabM Rule :=
  Lean.Elab.Command.liftTermElabM (parsePremise s)

-- This sets up the necessary context and state for CoreM.
def runCoreMFromIO (env : Lean.Environment) (x : Lean.CoreM α) : EIO Exception α := do
  -- Core.Context holds options, current namespace, open declarations, etc.
  -- We'll use default values for most, and the provided environment.
  let coreCtx : Lean.Core.Context := {
    fileName    := "<input>",
    fileMap     := default,
    options     := default,
    maxRecDepth := 1000, -- Default recursion depth
    maxHeartbeats := 0, -- No heartbeats limit for now
    currNamespace := `Server, -- Default namespace
    openDecls   := [Lean.OpenDecl.simple `Server []], -- No open declarations by default
    diag        := false -- Disable diagnostics for now, change to true for verbose
  }
  -- Core.State holds the environment, name generator, etc.
  let coreState : Lean.Core.State := {
    env := env,
    ngen := { namePrefix := `_run_core_ }
  }
  -- Run the CoreM monad
  let (result, _) ← x.run coreCtx coreState
  return result

def runMetaMFromIO (env : Lean.Environment) (x : Lean.MetaM α) : EIO Exception α := do
  runCoreMFromIO env <| x.run' {} {} -- MetaM.run' takes empty contexts/states

-- Helper to run TermElabM from IO
unsafe def runTermElabMFromIO (env : Environment) (x : Lean.Elab.TermElabM α) : IO α := do
  let metaMComp := x.run
  let res ← (runMetaMFromIO env metaMComp).toIO'
  match res with
  | Except.ok (result, _) =>
    pure result
  | Except.error err    => do
    let msg ← match err with
    | Exception.error ref msg =>
      let syntaxMsg ← (Lean.MessageData.ofSyntax ref).toString
      let msgStr ← msg.toString
      pure (s!"Syntax: '{syntaxMsg}' Message: '{msgStr}'")
    | Exception.internal id extra =>
      err.toMessageData.toString

    -- let msg ← err.toMessageData.toString
    throw (IO.userError msg)

-- Main evaluation server loop
unsafe def run_server : IO Unit := do
  IO.println "Loading experiences from experience.jsonl..."
  let experiences ← loadExperiences "experience.jsonl"
  IO.println s!"Loaded {experiences.length} experiences."

  -- Initialize Lean's environment for metaprogramming
  Lean.initSearchPath (← Lean.findSysroot)
  -- Import necessary modules for elaboration and evaluation.
  -- Std is needed for general utilities, especially if premises use things like `List.any`.
  let env ← Lean.importModules #[`Lean.Meta.Eval, `Lean.Elab.Command, `Lean.Parser, `Std, `Server.Common] {}

  IO.println "Starting evaluation server. Waiting for JSON input on stdin..."

  let stdin ← IO.getStdin
  -- Start the infinite loop for commands
  let rec loop : IO Unit := do
    let line ← stdin.getLine
    if line.trim.isEmpty then
      -- If an empty line is received, continue waiting, or break if desired.
      loop
    else
      match (Json.parse line >>= (fromJson? (α := EvaluationInput))) with
      | Except.ok inp =>
        let mut results : List EvaluationResult := []

        for premise_str in inp.premises do
          let rule ← runTermElabMFromIO env (parsePremise premise_str)
          let mut true_count := 0
          for exp in experiences do
            let (a, b) := moveUci exp.move
            IO.println s!"{a} {b} {rule exp.board a b} {exp.legal}"
            if (rule exp.board a b) == exp.legal then
              true_count := true_count + 1
          results := { premise := premise_str, true_count := true_count } :: results

        IO.println <| toJson results.reverse
        IO.println "" -- Newline signals end of response
        loop

      | Except.error err =>
        IO.eprintln s!"Error parsing command: {err} in line: {line}"
        IO.println <| toJson (Json.mkObj [("error", Json.str s!"Failed to parse command: {err}")])
        IO.println ""
        loop

  loop

end Server

unsafe def main (_ : List String) : IO Unit := do
  Server.run_server
