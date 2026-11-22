import Std

namespace SparseBitmapSharedLayout

theorem producerWordFitsMembershipWord (word bit : Nat) (hbit : bit < 32) :
    (word * 32 + bit) / 64 = word / 2 ∧
    (word * 32 + bit) % 64 = (word % 2) * 32 + bit := by
  omega

theorem laneHasSameHalfAsAbsoluteRow (tile lane : Nat) (hlane : lane < 32) :
    ((tile * 32 + lane) / 16) % 2 = lane / 16 := by
  omega

theorem halfPanelsPartitionRows (count : Nat) :
    count / 32 * 32 ≤ count / 16 * 16 ∧
    (count / 16 * 16 - count / 32 * 32 = 0 ∨
      count / 16 * 16 - count / 32 * 32 = 16) := by
  omega

theorem wordMergeAgreesWithAppendingBits (seen selected : Bool) :
    (if selected then true else seen) = (seen || selected) := by
  cases seen <;> cases selected <;> decide

end SparseBitmapSharedLayout
