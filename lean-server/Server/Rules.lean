import Server.Common

abbrev empty_1 (g : Grid) (agent : Coords) (c : Coords) : Bool :=
  g.get c = FIRE_3

abbrev empty_2 (g : Grid) (agent : Coords) (c : Coords) : Bool :=
  g.get c = FIRE_2 ∧ c = agent

abbrev empty_3 (g : Grid) (agent : Coords) (c : Coords) : Bool :=
  g.get c = FIRE_1 ∧ c = agent

abbrev tree_1 (g : Grid) (agent : Coords) (c : Coords) : Bool :=
  g.get c = EMPTY ∧ exists_foo c (fun c => g.get c = TREE)

abbrev fire1_1 (g : Grid) (agent : Coords) (c : Coords) : Bool :=
  g.get c = TREE ∧ exists_foo c (fun c => g.get c = FIRE_3)

abbrev fire2_1 (g : Grid) (agent : Coords) (c : Coords) : Bool :=
  g.get c = FIRE_1 ∧ c ≠ agent

abbrev fire3_1 (g : Grid) (agent : Coords) (c : Coords) : Bool :=
  g.get c = FIRE_2 ∧ c ≠ agent


abbrev empty (s : State) (c : Coords) : Bool :=
  empty_1 s.grid s.agent c ∨ empty_2 s.grid s.agent c ∨ empty_3 s.grid s.agent c

abbrev tree (s : State) (c : Coords) : Bool :=
  tree_1 s.grid s.agent c

abbrev fire1 (s : State) (c : Coords) : Bool :=
  fire1_1 s.grid s.agent c

abbrev fire2 (s : State) (c : Coords) : Bool :=
  fire2_1 s.grid s.agent c

abbrev fire3 (s : State) (c : Coords) : Bool :=
  fire3_1 s.grid s.agent c
