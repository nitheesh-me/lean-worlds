import Lean.Data.Json

open Lean
open Lean.Json

namespace Server

-- Existing code from the problem description
structure Coords where
  file : Int
  rank : Int
deriving DecidableEq, Repr, Inhabited, ToJson, FromJson

instance : ToString Coords where
  toString c := s!"[{c.file}, {c.rank}]"

inductive Owner where
| empty
| black
| white
deriving DecidableEq, Repr, Inhabited, ToJson, FromJson

inductive Piece where
| empty
| pawn
| rook
| knight
| bishop
| queen
| king
deriving DecidableEq, Repr, Inhabited, ToJson, FromJson

structure Board where
  data : Array (Array (Piece × Owner))
  piece (c : Coords) :=
    if 0 ≤ c.file ∧ c.file < 8 ∧ 0 ≤ c.rank ∧ c.rank < 8 then
      data[c.rank.toNat]![c.file.toNat]!.1
    else
      .empty
  owner (c : Coords) :=
    if 0 ≤ c.file ∧ c.file < 8 ∧ 0 ≤ c.rank ∧ c.rank < 8 then
      data[c.rank.toNat]![c.file.toNat]!.2
    else
      .empty
  deriving Inhabited

-- Manual ToJson/FromJson for Board due to custom methods
instance : ToJson Board where
  toJson b := toJson b.data

instance : FromJson Board where
  fromJson? j := do
    -- parse outer array of rows
    let rowArr ← j.getArr?
    -- for each row, parse array of cells
    let data ← rowArr.mapM fun rowJson => do
      let cellArr ← rowJson.getArr?
      -- for each cell, expect exactly [ pJson, oJson ]
      cellArr.mapM fun cellJson => do
        match cellJson with
        | Json.arr #[pJ, oJ] => do
          let p ← FromJson.fromJson? pJ
          let o ← FromJson.fromJson? oJ
          pure (p, o)
        | _ => throw s!"expected [p, o] for cell, got {cellJson}"
    -- build the Board
    pure { data := data }

def exists_coord (p : Int → Bool) : Bool :=
  (List.range 8).any (fun n => p n)

def distance (a b : Int) : Int :=
  Int.natAbs (a - b)

infixl:70 " distance " => distance
