import "div-helpers"
import "sub"
import "add"
import "mul"


--
-- Calculates (a * b) rem B^d
--
def multmod [m][ipb] (us: [ipb*(4*m)]u16) (vs: [ipb*(4*m)]u16) (d: i64) : [ipb*(4*m)]u16 = 
    let res = convMulV2 us vs
    in tabulate (ipb*(4*m)) (\i -> if i >= d then 0u16 else res[i])

--
-- Calculates B^h-v*w
--
def powDiff [m][ipb] (vs: [ipb*(4*m)]u16) (ws: [ipb*(4*m)]u16) (h: i64) (l: i64) : (u32, [ipb*(4*m)]u16) =
    let precV = prec vs
    let precW = prec ws
    let L = precW + precV - l + 1

    in if (precV == 0 || precW == 0) then
        let ret = zeroAndSet 1u16 h (ipb*(4*m))
        let ret = ret :> [ipb*(4*m)]u16
        in (1, ret)
    else if (L >= h) then
        let ret = convMulV2 vs ws
        in if ltBpow ret h then
            let bpow = zeroAndSet 1 h (ipb*(4*m))
            let (ret, _) = bsub bpow ret
            in (1, ret)
        else
            let bpow = zeroAndSet 1 h (ipb*(4*m))
            let (ret, _) = bsub ret bpow
            in (0, ret)
    else 
        let ret = multmod vs ws L
        in if !(ez ret) && ret[L-1] == 0 then 
            (0, ret)
        else 
            let bpow = zeroAndSet 1 L (ipb*(4*m))
            let (ret, _) = bsub bpow ret
            in (1, ret)

--
-- Iterate towards an approximation in at most log(M) steps
--
def step [m][ipb] (vs: [ipb*(4*m)]u16) (ws: [ipb*(4*m)]u16) (h: i64) (l: i64) (n: i64) : [ipb*(4*m)]u16 =
    let (sign, tmp) = powDiff ws vs (h-n) (l-2)
    let tmp = convMulV2 ws tmp
        |> shift (2 * n - h)
    let ws = shift n ws
    in if sign != 0 then
        --baddu16 ws tmp
        baddV3 ws tmp
    else
        -- bsubu16 ws tmp
        let (ret, _) = bsub tmp ws 
        in ret

--
-- Refine the approximation of the quotient
--
def refine [m][ipb] (vs: [ipb*(4*m)]u16) (ws: [ipb*(4*m)]u16) (h: i64) (k: i64) (l: i64) : [ipb*(4*m)]u16 =
    let ws = shift 2 ws
    let (ws, _, _) = loop (ws, l, i) = (ws, l, 0)
        while h - k > (l + 1) do
            let n = i64.min (h - k + 1 - l) l
            let s = i64.max 0 (k - 2 * l + 1 - 2)
            let vs = shift (-s) vs
            let tmp = step vs ws (k + l + n - s + 2) l n
            let ws = shift (-1) tmp 
            let l = l + n - 1 
            let i = i + 1
            in (ws, l, i)
    in shift (-2) ws

--
-- Calculates the shifted inverse
--
def shinv [m][ipb] (vs: [ipb*(4*m)]u16) (h: i64) (k: i64) : [ipb*(4*m)]u16 =
    if k == 0 then
        quo_single h (vs) (ipb*(4*m)) :> [ipb*(4*m)]u16 
    else if k >= h && !(eqBpow vs h) then
        vs
    else if k == h - 1 && vs[k] > u16.highest / 2 then
        zeroAndSet 1 0 (ipb*(4*m)) :> [ipb*(4*m)]u16
    else if eqBpow vs k then
        zeroAndSet 1 (h - k) (ipb*(4*m)) :> [ipb*(4*m)]u16
    else 
        let l = i64.min k 2
        let V = (u64.u16 vs[k - 2]) | ((u64.u16 vs[k - 1]) << 1*16) | ((u64.u16 vs[k]) << 2*16)
        let b2l = 1u64 << 4*16
        let tmp = (b2l - V) / (V + 1)

        let ws = tabulate (ipb*(4*m)) (\i -> 
            if i == 0 then u16.u64 tmp
            else if i == 1 then u16.u64 (tmp >> 16)
            else 0u16 )
        in if h - k < l then
            shift (h-k-l) ws
        else
            refine vs ws h k l

--
-- Implementation of multi-precision integer division using
-- the shifted inverse and classical multiplication
--
def div [m][ipb] (us: [ipb*(4*m)]u16) (vs: [ipb*(4*m)]u16) : ([ipb*(4*m)]u16, [ipb*(4*m)]u16) =
    let h = prec us
    let k = (prec vs) - 1

    let (kIsOne, us, vs, h, k) =
        if k == 1 then
            let h = h + 1
            let k = k + 1
            let us = shift 1 us
            let vs = shift 1 vs
            in (true, us, vs, h, k)
        else
            (false, us, vs, h, k)

    let quo = 
        let m = m * 2
        let quo_padded = ((shinv vs h k) ++ (replicate (ipb*(4*(m/2))) 0u16)) :> [ipb * (4 * m)]u16
        let us_padded = (us ++ (replicate (ipb*(4*(m/2))) 0u16)) :> [ipb * (4 * m)]u16
        let mul_res = convMulV2 quo_padded us_padded
        let mul_shifted = shift (-h) mul_res
        let res = take (ipb*(4*(m/2))) mul_shifted
        in res
    let quo = quo :> [ipb * (4 * m)]u16

    let (rem, _) = convMulV2 vs quo
        |> bsub us

    let (quo, rem) =
        if not (lt rem vs) then
            let quo = badd1u16 quo
            let (rem, _) = bsub rem vs
            in (quo, rem)
        else
            (quo, rem)

    let rem =
        if kIsOne then
            shift (-1) rem
        else 
            rem

    in (quo, rem)

--
-- Implementation of multi-precision integer quotient using
-- the shifted inverse and classical multiplication
--
def quo [m][ipb] (us: [ipb*(4*m)]u16) (vs: [ipb*(4*m)]u16) : [ipb*(4*m)]u16 =
    let h = prec us
    let k = (prec vs) - 1

    let (us, vs, h, k) =
        if k == 1 then
            let h = h + 1
            let k = k + 1
            let us = shift (-1) us
            let vs = shift (-1) vs
            in (us, vs, h, k)
        else
            (us, vs, h, k)

   let quo = 
        let m = m * 2
        let quo_padded = ((shinv vs h k) ++ (replicate (ipb*(4*(m/2))) 0u16)) :> [ipb * (4 * m)]u16
        let us_padded = (us ++ (replicate (ipb*(4*(m/2))) 0u16)) :> [ipb * (4 * m)]u16
        let mul_res = convMulV2 quo_padded us_padded
        let mul_shifted = shift (-h) mul_res
        let res = take (ipb*(4*(m/2))) mul_shifted
        in res
    let quo = quo :> [ipb * (4 * m)]u16

    let rem = convMulV2 vs quo
        |> bsubu16 us

    let quo =
        if not (lt rem vs) then
            badd1u16 quo
        else
            quo

    in quo


-- testing division
-- 
-- entry: test_div
-- compiled input { [20u16, 42u16, 10u16, 4u16, 63u16, 8u16, 22u16, 1u16] [5u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16] }
-- output { [4u16, 39330u16, 39323u16, 52429u16, 13119u16, 39323u16, 13111u16, 0u16] [0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16] }
-- compiled input { [0u16,1u16,4u16,0u16] [420u16, 0u16, 0u16, 0u16]}
-- output { [10142u16, 624u16, 0u16, 0u16] [200u16, 0u16, 0u16, 0u16] }
-- compiled input { [4u16,2u16,2u16,2u16,0u16,0u16,0u16,0u16] [4u16,1u16,1u16,1u16,0u16,0u16,0u16,0u16] }
-- output { [1u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16] [0u16, 1u16, 1u16, 1u16, 0u16, 0u16, 0u16, 0u16] }
-- compiled input { [1u16,4u16,2u16,3u16] [0u16,4u16,1u16,0u16] }
-- output { [65526u16, 2u16, 0u16, 0u16] [1u16, 44u16, 0u16, 0u16] }
-- compiled input { [7u16, 3u16, 5u16, 10u16, 7u16, 9u16, 7u16, 9u16, 7u16, 2u16, 2u16, 10u16, 0u16, 0u16, 0u16, 0u16] [4u16, 4u16, 5u16, 1u16, 7u16, 2u16, 2u16, 10u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16] } 
-- output { [0u16, 0u16, 0u16, 0u16, 1u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16] [7u16, 3u16, 5u16, 10u16, 3u16, 5u16, 2u16, 8u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16] }
-- compiled input { [4u16, 2u16, 3u16, 6u16, 9u16, 6u16, 10u16, 10u16, 10u16, 9u16, 9u16, 9u16, 0u16, 0u16, 0u16, 0u16] [10u16, 1u16, 1u16, 5u16, 10u16, 2u16, 4u16, 4u16, 1u16, 1u16, 1u16, 0u16, 0u16, 0u16, 0u16, 0u16] }
-- output { [65535u16, 8u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16] [14u16, 65449u16, 65530u16, 1u16, 65510u16, 65453u16, 65531u16, 65513u16, 65510u16, 0u16, 1u16, 0u16, 0u16, 0u16, 0u16, 0u16] }
-- compiled input { [39017u16, 18547u16, 56401u16, 23807u16, 37962u16, 22764u16, 7977u16, 31949u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16] [22714u16, 55211u16, 16882u16, 7931u16, 43491u16, 57670u16, 124u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16] }
-- output { [54959u16, 255u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16] [30019u16, 15584u16, 62297u16, 30007u16, 47579u16, 3678u16, 23u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16] }
-- compiled input { [39017u16, 18547u16, 56401u16, 23807u16, 37962u16, 22764u16, 7977u16, 31949u16, 22714u16, 55211u16, 16882u16, 7931u16, 43491u16, 57670u16, 124u16, 25282u16, 2132u16, 10232u16, 8987u16, 59880u16, 52711u16, 17293u16, 3958u16, 9562u16, 63790u16, 29283u16, 49715u16, 55199u16, 50377u16, 1946u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16]
-- [64358u16, 23858u16, 20493u16, 55223u16, 47665u16, 58456u16, 12451u16, 55642u16, 24869u16, 35165u16, 45317u16, 41751u16, 43096u16, 23273u16, 33886u16, 43220u16, 48555u16, 36018u16, 53453u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16] }
-- output { [19472u16, 62163u16, 27479u16, 13589u16, 47175u16, 43963u16, 55342u16, 58871u16, 55235u16, 53043u16, 2386u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16]
-- [39433u16, 45455u16, 53114u16, 8163u16, 2139u16, 41117u16, 26901u16, 18168u16, 43904u16, 52648u16, 42003u16, 21686u16, 4014u16, 49277u16, 30849u16, 40590u16, 42920u16, 59996u16, 43580u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16] }
entry test_div  [m] (us: [m]u16) (vs: [m]u16) : ([m]u16, [m]u16) =
    let mdiv4 = m / 4
    let us = us :> [1*(4*mdiv4)]u16
    let vs = vs :> [1*(4*mdiv4)]u16
    let ret = div us vs :> ([m]u16, [m]u16)
    in ret

-- 
-- entry: test_quo
-- compiled input { [39017u16, 18547u16, 56401u16, 23807u16, 37962u16, 22764u16, 7977u16, 31949u16, 22714u16, 55211u16, 16882u16, 7931u16, 43491u16, 57670u16, 124u16, 25282u16, 2132u16, 10232u16, 8987u16, 59880u16, 52711u16, 17293u16, 3958u16, 9562u16, 63790u16, 29283u16, 49715u16, 55199u16, 50377u16, 1946u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16]
-- [64358u16, 23858u16, 20493u16, 55223u16, 47665u16, 58456u16, 12451u16, 55642u16, 24869u16, 35165u16, 45317u16, 41751u16, 43096u16, 23273u16, 33886u16, 43220u16, 48555u16, 36018u16, 53453u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16] }
-- output { [19472u16, 62163u16, 27479u16, 13589u16, 47175u16, 43963u16, 55342u16, 58871u16, 55235u16, 53043u16, 2386u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16, 0u16] }
entry test_quo (us: [1*(4*16)]u16) (vs: [1*(4*16)]u16) : [1*(4*16)]u16 =
    quo us vs

-- 
-- entry: bench_div
-- compiled random input { [65536][64]u16   [65536][64]u16  }
-- compiled random input { [32768][128]u16  [32768][128]u16 }
-- compiled random input { [16384][256]u16  [16384][256]u16 }
-- compiled random input { [8192][512]u16   [8192][512]u16  }
-- compiled random input { [4096][1024]u16  [4096][1024]u16 }
-- compiled random input { [2048][2048]u16  [2048][2048]u16 }
-- compiled random input { [1024][4096]u16  [1024][4096]u16 }
-- compiled random input { [512][8192]u16   [512][8192]u16  }
-- compiled random input { [256][16384]u16  [256][16384]u16 }
entry bench_div [n][ipb][m] (us: [n][ipb][m]u16) (vs: [n][ipb][m]u16) : ([n][ipb*(4*m)]u16, [n][ipb*(4*m)]u16) =
    let mdiv4 = m / 4
    let us = (map flatten us) :> [n][ipb*(4*mdiv4)]u16
    let vs = (map flatten vs) :> [n][ipb*(4*mdiv4)]u16
    let ret = map2 div us vs |> unzip :> ([n][ipb*(4*m)]u16, [n][ipb*(4*m)]u16)
    in  ret

-- 
-- entry: bench_quo
-- compiled random input { [65536][2][64]u16   [65536][2][64]u16  }
-- compiled random input { [32768][2][128]u16  [32768][2][128]u16 }
-- compiled random input { [16384][2][256]u16  [16384][2][256]u16 }
-- compiled random input { [8192][2][512]u16   [8192][2][512]u16  }
-- compiled random input { [4096][2][1024]u16  [4096][2][1024]u16 }
-- compiled random input { [2048][2][2048]u16  [2048][2][2048]u16 }
-- compiled random input { [1024][2][4096]u16  [1024][2][4096]u16 }
-- compiled random input { [512][2][8192]u16   [512][2][8192]u16  }
-- compiled random input { [256][2][16384]u16  [256][2][16384]u16 }
entry bench_quo [n][ipb][m] (us: [n][ipb][m]u16) (vs: [n][ipb][m]u16) : [n][ipb*(4*m)]u16 =
    #[unsafe]
    let mdiv4 = m / 4
    let us = (map flatten us) :> [n][ipb*(4*mdiv4)]u16
    let vs = (map flatten vs) :> [n][ipb*(4*mdiv4)]u16
    let ret = map2 quo us vs :> [n][ipb*(4*m)]u16
    in  ret

-- 
-- entry: bench_div_single
-- compiled random input { [64]u16    [64]u16    }
-- compiled random input { [128]u16   [128]u16   }
-- compiled random input { [256]u16   [256]u16   }
-- compiled random input { [512]u16   [512]u16   }
-- compiled random input { [1024]u16  [1024]u16  }
-- compiled random input { [2048]u16  [2048]u16  }
-- compiled random input { [4096]u16  [4096]u16  }
-- compiled random input { [8192]u16  [8192]u16  }
-- compiled random input { [16384]u16 [16384]u16 }
-- compiled random input { [32768]u16 [32768]u16 }
-- compiled random input { [65536]u16 [65536]u16 }
-- compiled random input { [131072]u16 [131072]u16 }
-- compiled random input { [262144]u16 [262144]u16 }
-- compiled random input { [524288]u16 [524288]u16 }
entry bench_div_single [m] (us: [m]u16) (vs: [m]u16) : ([m]u16, [m]u16) =
    let mdiv4 = m / 4
    let us = us :> [1*(4*mdiv4)]u16
    let vs = vs :> [1*(4*mdiv4)]u16
    in div us vs :> ([m]u16, [m]u16)

-- bench_div_replicated, is possible not to give th most accurate beching results.
-- Since we replicated the inputs, they will all fall into the same fast or slow case of the algortihm 
-- resulting the the bench marking varying based on the chosen random input.
-- however only way we have managed to get batchd beching to work, with running out of memory.

-- 
-- entry: bench_div_replicated
-- compiled random input { [64]u16    [64]u16    }
-- compiled random input { [128]u16   [128]u16   }
-- compiled random input { [256]u16   [256]u16   }
-- compiled random input { [512]u16   [512]u16   }
-- compiled random input { [1024]u16  [1024]u16  }
-- compiled random input { [2048]u16  [2048]u16  }
-- compiled random input { [4096]u16  [4096]u16  }
-- compiled random input { [8192]u16  [8192]u16  }
entry bench_div_replicated [m] (us: [m]u16) (vs: [m]u16) : ([][m]u16, [][m]u16) =
    let mdiv4 = m / 4
    let instances = 134217728 / m
    let us = (us :> [1*(4*mdiv4)]u16) |> replicate instances
    let vs = (vs :> [1*(4*mdiv4)]u16) |> replicate instances
    in map2 div us vs |> unzip :> ([instances][m]u16, [instances][m]u16)
