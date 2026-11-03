# Recommendations

The following is a list of recommendations and best practices we have established at the [LEGO Island Decompilation project](https://github.com/isledecomp/isle). They do not affect the output of `reccmp`, so it is up to you if you want to use them.

## Class/struct size annotation and assertion

Once we have a reasonable guess for the size of a class or struct, we add it in a comment like so:

```c++
// SIZE 0x1c
class MxCriticalSection {
public:
    MxCriticalSection();
    ~MxCriticalSection();
    static void SetDoMutex();
    // ...
}
```

Furthermore, we use a compile-time assertion to verify that the recompiled size is correct (see also [this file](https://github.com/isledecomp/isle/blob/82453f62d84f979f8a6fc7b46e21b61cb835d2f1/util/decomp.h)):

```c++
#define DECOMP_STATIC_ASSERT(V)       \
    namespace                         \
    {                                 \
    typedef int foo[(V) ? 1 : -1];    \
    }
#define DECOMP_SIZE_ASSERT(T, S) DECOMP_STATIC_ASSERT(sizeof(T) == S)
```

Then we add `DECOMP_SIZE_ASSERT(MxCriticalSection, 0x1c)` to the respective `.cpp` file (if the class has no dedicated `.cpp` file, we use any appropriate `.cpp` file where the class is used).

## Member variables

We annotate member variables with their relative offsets.

```c++
class MxDSObject : public MxCore {
private:
    MxU32 m_sizeOnDisk;   // 0x08
    MxU16 m_type;         // 0x0c
    char* m_sourceName;   // 0x10
    undefined4 m_unk0x14; // 0x14
    // ...
}
```

## VTable members

In addition to the `VTABLE` annotation (which is relevant to `reccmp`), we also add comments to indicate the relative offset of each function:

```c++
// VTABLE: LEGO1 0x100dc900
class MxEventManager : public MxMediaManager {
public:
    MxEventManager();
    virtual ~MxEventManager() override;

    virtual void Destroy() override;                                     // vtable+0x18
    virtual MxResult Create(MxU32 p_frequencyMS, MxBool p_createThread); // vtable+0x28
    // ...
}
```

## Aliases for unknown scalar types

In order to distinguish known from unknown types, we have added the following typedefs:

```c++
typedef unsigned char undefined;
typedef unsigned short undefined2;
typedef unsigned int undefined4;
```

Note that the behaviour of signed and unsigned integers can be different even when no arithmetic is involved. If changing e.g. from `undefined4` to `int` improves the match, this is a strong indicator that the original variable was signed as well.

## Calls routed through a wrapper function

Some projects cannot emit a direct call to an original function and instead
generate a wrapper (often a template instantiation, e.g. a `FunctionResolver`)
that performs the call. The recomp assembly then reads

```
0x4b21db : -call <OFFSET12>
         : +call FunctionResolver::Resolver<void (__thiscall A::B::*)(int),0,4657152,&A::B::f,0>::GameFunction<...>::CallHelper<void,void>::call (FUNCTION)
```

Every such call counts as a mismatch, so an otherwise perfect function can
never reach 100%. `reccmp-reccmp` offers two options for this situation:

* `--resolve-wrapped-calls` uses the fact that the original address of the
  called function appears inside the wrapper's name, because that is how the
  wrapper is parameterized (`4657152` == `0x471000` above). If that number is
  the target address of the call on the original side, the two instructions
  describe the same call and the recomp instruction is rewritten to the text
  used by the original one. Calls that go somewhere else are left alone, so
  genuinely wrong calls are still reported. Both decimal and hexadecimal
  (`0x...`) numbers are recognized, and numbers below `0x1000` are ignored so
  that small template parameters are not mistaken for an address.
* `--ignore-call-targets` is the blunt alternative: the target of every call is
  replaced by a `<CALL>` placeholder in both binaries, so call targets never
  contribute to the difference. This also hides real call mismatches, so prefer
  `--resolve-wrapped-calls` where it works. If both options are given,
  `--ignore-call-targets` wins.

Neither option is enabled by default.

## The stack security cookie (`/GS`)

A function compiled with `/GS` that needs a stack guard (in practice, any
function with a local array) reads the global security cookie in its prologue
and validates it in its epilogue:

```
mov eax, dword ptr [__security_cookie]
xor eax, esp
mov dword ptr [esp + 0x1c], eax
...
mov ecx, dword ptr [esp + 0x1c]
xor ecx, esp
call __security_check_cookie
```

The cookie global and the check function come from the compiler and the CRT,
not from the source, so the recomp binary has its own copy of both at its own
address while the original reads its own. reccmp names each side after the
entity it finds at that address, and the diff reads

```
-mov eax, dword ptr [StructResolver::Instance<unsigned int,12141088>::instance (DATA)]
+mov eax, dword ptr [__security_cookie (DATA)]
```

Nothing is wrong with the recompiled code, but the function cannot reach 100%.

`--ignore-security-cookie` replaces the cookie operand with a
`<SECURITY_COOKIE>` placeholder, and the target of the check call with a
`<SECURITY_CHECK_COOKIE>` placeholder, in **both** binaries, so those operands
never contribute to the difference. The option is not enabled by default.

The idiom is recognized by its shape rather than by the name of the symbol,
because the two binaries do not agree on that name:

* a load from a named memory location immediately followed by
  `xor <the same register>, esp` (or `ebp`) is the cookie read;
* a direct call immediately preceded by `xor <register>, esp` (or `ebp`) is
  the call to the cookie check function.

Only the operand is replaced. The instructions themselves, the register used
and the stack slot the cookie is written to all still count, so a function
that gets its stack guard wrong is still reported as a mismatch. A global that
is read but not mixed with the stack pointer is left alone.

An alternative is to make the recompiled binary use the original's cookie: in
MASM, define `__security_cookie`, `__security_cookie_complement` and a stubbed
`__security_init_cookie` at the original addresses so that the CRT's
`gs_cookie.obj` is never linked. That changes the binary you ship, while this
option changes only the comparison.
