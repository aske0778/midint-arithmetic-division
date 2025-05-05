--
-- Returns the precision of a bigint eg. the number of leading zeros
--
def prec [n] (u : [n]u16) : (i64) = 
    let bar = reduce (\idx1 idx2 -> 
                        if u[idx2] != 0 then (idx2 + 1) else idx1) (0i64) (iota n)
    in bar
--
-- Checks if two bigints are equal
--
def eq [n] (u : [n]u16) (v : [n]u16) : bool =
    map2 (==) u v |> reduce (&&) true

--
-- Checks if bigint is zero
--
def ez [n] (us : [n]u16) : bool = 
    all (== 0) us

--
-- Checks if u < v
--
def lt [m][ipb] (us: [ipb*(4*m)]u16) (vs: [ipb*(4*m)]u16) : bool =
  let res = map2 (\ x y -> (x < y, x == y) ) us vs
  |> reduce (\ (l1, e1) (l2, e2) -> (l2 || (e2 && l1), e1 && e2) ) (false, true)
  in res.0

--
-- Sets a given index of the bigint to d in place
--
def set [n] (u : *[n]u16) (d : u16) (idx : i32) : [n]u16 = 
    let u[idx] =  d in u

--
-- Returns a bigint where the ith index is set to d
--
def zeroAndSet (d : u16) (idx : i64) (m : i64) : [m]u16 = 
    tabulate m (\i -> if i == idx then d else 0u16)

-- --
-- -- Checks if u < b where b is precision of bpow
-- --    
-- def ltbpow [m][ipb] (u: [ipb*(4*m)]u16) (bpow: i64) : bool =
--   let v = zeroAndSet 1 bpow (ipb*(4*m)) :> [ipb*(4*m)]u16
--   let res = map2 (\ x y -> (x < y, x == y) ) u v
--   |> reduce (\ (l1, e1) (l2, e2) -> (l2 || (e2 && l1), e1 && e2) ) (false, true)
--   in res.0

--
-- Shifts the bigint by n either left or right
--
def shift [m][ipb] (shft : i64) (u : [ipb*(4*m)]u16) : ([ipb*(4*m)]u16) =
    map (\idx ->
            let offset = idx - shft 
            in if offset < (ipb*(4*m)) && offset >= 0 then u[offset] else 0) (iota (ipb*(4*m)))

--
-- Checks equality of u and bpow
--
def eqBpow [m][ipb] (u : [ipb*(4*m)]u16) (b : i64) : bool =
    let bpow = zeroAndSet 1 b (ipb*(4*m)) :> [ipb*(4*m)]u16
    in map2 (==) u bpow |> reduce (&&) true

--
-- Checks equality of u and bpow
--
def ltBpow [m][ipb] (u: [ipb*(4*m)]u16) (b: i64) : bool =
    let bpow = zeroAndSet 1 b (ipb*(4*m)) :> [ipb*(4*m)]u16
    in lt u bpow

--
-- Calculates quo on bpow and digit d
--
def quo_single [m][ipb] (bpow : i64) (d :[ipb*(4*m)]u16) (n : i64) : ([ipb*(4*m)]u16) =
    let ret = replicate n 0u16 :> [ipb*(4*m)]u16
    let (_r,ret) = loop (r,ret) = (1u32, copy ret) for i < bpow do
                let r = r << 16
                in
                if (r > (u32.u16 d[0])) then 
                                let ret[(bpow - 1) - i] = u16.u32(r / (u32.u16 d[0]))
                                in
                                (r % (u32.u16 d[0]), ret)
                        else 
                                (r, ret)
    in ret


-- Testing ez
-- ==
-- entry: ez_test 
-- compiled input { [0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32] }
-- output { true }
-- compiled input { [1u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32] }
-- output { false }
-- compiled input { [0u32, 0u32, 0u32, 1u32, 0u32, 0u32, 0u32, 0u32] }
-- output { false }
-- compiled input { [0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 1u32] }
-- output { false }

-- Testing lt 
-- == 
-- entry: lt_test
-- compiled input {   [1u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32]
--                    [2u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32]}
-- output { true }
-- compiled input {   [2u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32]
--                    [1u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32]}
-- output { false }
-- compiled input {   [2u32, 0u32, 1u32, 0u32, 0u32, 0u32, 0u32, 0u32]
--                    [1u32, 0u32, 0u32, 1u32, 0u32, 0u32, 0u32, 0u32]}
-- output { true }
-- compiled input {   [2u32, 0u32, 1u32, 0u32, 0u32, 1u32, 0u32, 0u32]
--                    [1u32, 0u32, 0u32, 1u32, 0u32, 0u32, 0u32, 0u32]}
-- output { false }
-- compiled input {   [2u32, 0u32, 1u32, 0u32, 0u32, 1u32, 0u32, 0u32]
--                    [1u32, 0u32, 0u32, 1u32, 0u32, 0u32, 0u32, 1u32]}
-- output { true }

-- Testing eq
-- == 
-- entry: eq_test
-- compiled input {   [1u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32]
--                    [2u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32]}
-- output { false }
-- compiled input {   [1u32, 0u32, 1u32, 0u32, 1u32, 0u32, 0u32, 1u32]
--                    [1u32, 0u32, 1u32, 0u32, 1u32, 0u32, 0u32, 1u32]}
-- output { true }
-- compiled input {   [2u32, 0u32, 1u32, 0u32, 0u32, 0u32, 0u32, 0u32]
--                    [1u32, 0u32, 0u32, 1u32, 0u32, 0u32, 0u32, 0u32]}
-- output { false }
-- compiled input {   [2u32, 0u32, 1u32, 0u32, 0u32, 1u32, 0u32, 0u32]
--                    [1u32, 0u32, 0u32, 1u32, 0u32, 0u32, 0u32, 0u32]}
-- output { false }
-- compiled input {   [2u32, 0u32, 1u32, 1u32, 0u32, 1u32, 0u32, 1u32]
--                    [2u32, 0u32, 1u32, 1u32, 0u32, 1u32, 0u32, 1u32]}
-- output { true }


--                    [2u32,  0u32,  1u32,  0u32, 0u32,  1u32,  0u32,  0u32]
--                    [1u32,  0u32,  0u32,  1u32, 0u32,  0u32,  0u32,  1u32]
--                    [false, false, false, true, false, false, false, true]

--                    [true, false, false, false, true, false, false, false]

--                    [2u32,  0u32,  1u32,  0u32, 0u32,  1u32,  1u32,  1u32]
--                    [1u32,  0u32,  0u32,  1u32, 0u32,  0u32,  2u32,  1u32]
--                    [false, false, false, true, false, false, true, false]