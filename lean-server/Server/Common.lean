import Lean.Data.Json

open Lean
open Lean.Json

def GRID_SIZE : Nat := 10

/--
0: empty
1: tree
2: fire_1
3: fire_2
4: fire_3
5: rock
-/
structure Grid where
  data : Array (Array Nat)
  deriving Inhabited, ToJson, FromJson, Repr

structure Coords where
  row : Nat
  col : Nat
  deriving Inhabited, DecidableEq, ToJson, FromJson, Repr

structure State where
  grid : Grid
  agent : Coords
  deriving Inhabited, ToJson, FromJson, Repr

abbrev EMPTY := 0
abbrev TREE := 1
abbrev FIRE_1 := 2
abbrev FIRE_2 := 3
abbrev FIRE_3 := 4
abbrev ROCK := 5

def Grid.get (g : Grid) (c : Coords) : Nat :=
  g.data[c.row]![c.col]!

def within_bounds (i : Nat) : Bool :=
  i >= 0 ∧ i < GRID_SIZE

def exists_foo (c : Coords) (h : Coords → Bool) : Bool :=
  let neighbors : List (Nat × Nat) :=
    [ (c.row - 1, c.col), (c.row + 1, c.col), (c.row, c.col - 1), (c.row, c.col + 1) ]
  neighbors.any (fun (r, c) => within_bounds r ∧ within_bounds c ∧ h ⟨r, c⟩)
