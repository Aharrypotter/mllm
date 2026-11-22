import Lean

set_option autoImplicit false
set_option warningAsError true

namespace ShadowNPU.SparsePvLayout

-- The experimental CPU kernel stores 16 FP16 weights per key and writes 128 output channels per row.
def panelOffset (key row : Nat) : Nat := key * 16 + row
def outputOffset (row channel : Nat) : Nat := row * 128 + channel
def tiledEnd (begin count : Nat) : Nat := begin + count / 16 * 16

theorem panel_address (key row keyCount : Nat) (hKey : key < keyCount) (hRow : row < 16) :
    panelOffset key row < keyCount * 16 ∧
      panelOffset key row / 16 = key ∧ panelOffset key row % 16 = row := by
  unfold panelOffset
  omega

theorem panel_injective (a rowA b rowB : Nat) (hA : rowA < 16) (hB : rowB < 16)
    (same : panelOffset a rowA = panelOffset b rowB) : a = b ∧ rowA = rowB := by
  unfold panelOffset at same
  omega

theorem row_partition (begin count : Nat) :
    begin ≤ tiledEnd begin count ∧ tiledEnd begin count ≤ begin + count ∧
      begin + count - tiledEnd begin count = count % 16 := by
  unfold tiledEnd
  omega

-- The pointer recovered by subtracting the last row position is the tile's first output row.
theorem writeback_base (begin tile localRow : Nat) :
    outputOffset (begin + tile * 16 + localRow) 0 - localRow * 128 =
      outputOffset (begin + tile * 16) 0 := by
  unfold outputOffset
  omega

theorem tile_output_in_bounds (begin count tile localRow channel : Nat)
    (hTile : tile < count / 16) (hRow : localRow < 16) (hChannel : channel < 128) :
    outputOffset begin 0 ≤ outputOffset (begin + tile * 16 + localRow) channel ∧
      outputOffset (begin + tile * 16 + localRow) channel < outputOffset (tiledEnd begin count) 0 := by
  unfold outputOffset tiledEnd
  omega

theorem split_worker_writes_disjoint (leftRow leftChannel rightRow rightChannel split : Nat)
    (hLeft : leftRow < split) (hRight : split ≤ rightRow)
    (hLeftChannel : leftChannel < 128) (hRightChannel : rightChannel < 128) :
    outputOffset leftRow leftChannel ≠ outputOffset rightRow rightChannel := by
  unfold outputOffset
  omega

-- The packed HVX mask stores four row words for each group of 32 causal columns.
def maskWord (key row : Nat) : Nat := key / 32 * 4 + row

theorem hvx_visible_address (firstVisible key row : Nat) (hRow : row < 4) (hKey : key < firstVisible + row) :
    maskWord key row < ((firstVisible + 3 + 31) / 32) * 4 ∧
      maskWord key row % 4 = row ∧ maskWord key row / 4 * 32 + key % 32 = key := by
  unfold maskWord
  omega

theorem page_address (key : Nat) : key / 64 * 64 + key % 64 = key := by omega

-- Integer arithmetic describes which contributions may be omitted. Rounding and instructions are outside this model.
def pv (keys : List Nat) (weight value : Nat → Int) : Int :=
  keys.foldr (fun key acc => weight key * value key + acc) 0

theorem pv_cons (key : Nat) (rest : List Nat) (weight value : Nat → Int) :
    pv (key :: rest) weight value = weight key * value key + pv rest weight value := rfl

theorem zero_weight_filter (keys : List Nat) (active : Nat → Bool) (weight value : Nat → Int)
    (omitted : ∀ key, active key = false → weight key = 0) :
    pv (keys.filter active) weight value = pv keys weight value := by
  induction keys with
  | nil => rfl
  | cons key rest ih =>
    cases used : active key with
    | false => simp [List.filter_cons, used, pv_cons, omitted key used, ih]
    | true => simp [List.filter_cons, used, pv_cons, ih]

end ShadowNPU.SparsePvLayout

#print axioms ShadowNPU.SparsePvLayout.panel_address
#print axioms ShadowNPU.SparsePvLayout.panel_injective
#print axioms ShadowNPU.SparsePvLayout.row_partition
#print axioms ShadowNPU.SparsePvLayout.writeback_base
#print axioms ShadowNPU.SparsePvLayout.tile_output_in_bounds
#print axioms ShadowNPU.SparsePvLayout.split_worker_writes_disjoint
#print axioms ShadowNPU.SparsePvLayout.hvx_visible_address
#print axioms ShadowNPU.SparsePvLayout.page_address
#print axioms ShadowNPU.SparsePvLayout.zero_weight_filter
