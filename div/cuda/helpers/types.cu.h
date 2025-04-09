#define WARP   (32)
#define lgWARP  (5)

#define HIGHEST32 ( 0xFFFFFFFF )
#define HIGHEST64 ( 0xFFFFFFFFFFFFFFFF )

typedef unsigned __int128 uint128_t;

struct U64bits {
    using uint_t = uint64_t;
    using sint_t = int64_t;
    using ubig_t = uint128_t;
    using uquad_t = uint128_t;
    using carry_t= uint32_t;
    static const int32_t  bits = 64;
    static const uint_t HIGHEST = 0xFFFFFFFFFFFFFFFF;
};

struct U32bits {
    using uint_t = uint32_t;
    using sint_t = int32_t;
    using ubig_t = uint64_t;
    using uquad_t = uint128_t;
    using carry_t= uint32_t;
    static const int32_t  bits = 32;
    static const uint_t HIGHEST = 0xFFFFFFFF;
};

struct U16bits {
    using uint_t = uint16_t;
    using sint_t = int16_t;
    using ubig_t = uint32_t;
    using uquad_t = uint64_t;
    using carry_t= uint16_t;
    static const int16_t  bits = 16;
    static const uint_t HIGHEST = 0xFFFF;
};

#define LIFT_LEN(m,q) (((m + q - 1) / q) * q)

template<class CT>
class CarrySegBop {
  public:
    typedef CT InpElTp;
    typedef CT RedElTp;
    static const bool commutative = false;
    static __device__ __host__ inline CT identInp()           { return (CT)0; }
    static __device__ __host__ inline CT mapFun(const CT& el) { return el;    }
    static __device__ __host__ inline CT identity()           { return (CT)2; }

    static __device__ __host__ inline 
    CT apply(const CT c1, const CT c2) {
        CT res;
        if (c2 & 4) {
            res = c2;
        } else {
            res = ( (c1 & (c2 >> 1)) | c2 ) & 1;
            res = res | (c1 & c2  & 2);
            res = res | ( (c1 | c2) & 4 );
        }
        return res;
    }
    static __device__ __host__ inline bool equals(const CT t1, const CT t2) { return (t1 == t2); }
    static __device__ __host__ inline CT remVolatile(volatile CT& t)   { CT res = t; return res; }
};

class LessThan {
  public:
    static __device__ inline int apply(int elm, int u) { 
        u |= (u & 0b10) && (elm & 0b01);
        u = (u & ~0b10) | (u & elm & 0b10);
        return u;
    }
};
