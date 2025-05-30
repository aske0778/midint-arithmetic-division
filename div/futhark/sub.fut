import "helper"
import "add"

-- Subtraction libary develop by Thorbjørn Bülow Bringgaard for his master thesis "Efficient Big Integer Arithmetic Using GPGPU"
-- GitHub repositry: https://github.com/tossenxD/big-int
-- used as a framework during this thesis
-- to provide the needed functionallity to make an implementation of division on multi-precision integers 

--------------------------------------------------------------------------------
--- Big Integer Subtraction
--------------------------------------------------------------------------------
--- Defined from addition, see `add.fut` for more information.
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
-- Based on baddV1; false is `+`, true is `-`
--------------------------------------------------------------------------------

def bsub [m] (us: [m]ui) (vs: [m]ui) : ([m]ui, bool) =
  -- 1. compute sign
  let (gs, ls, sign) = if lt us vs then (vs, us, true) else (us, vs, false)

  -- 2. compute absolute sums and carries
  let (ws, cs) = unzip <| imap2 gs ls
    (\ g l -> let w = g-l in (w, (boolToCt (w > g)) | ((boolToCt (w==0)) << 1)))

  -- 3. propagate carries
  let pcs = scanExc carryProp carryPropE cs

  -- 4. add carries to sums
  in (map2 (\ w c -> w - fromCt (c & 1)) ws pcs, sign)
