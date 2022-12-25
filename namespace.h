// Customize namespace prefix by defining the macro KS_NAMESPACE.
// Disable namespace prefix by setting KS_NO_NAMESPACE=1.

#pragma once

#ifndef KS_NAMESPACE
#define KS_NAMESPACE ks
#endif

#if KS_NO_NAMESPACE
#undef KS_NAMESPACE
#endif

// clang-format off
#ifdef KS_NAMESPACE
#define KS_NAMESPACE_BEGIN namespace KS_NAMESPACE {
#define KS_NAMESPACE_END }
#define USING_NAMESPACE_KS using namespace ks;
#else
#define KS_NAMESPACE_BEGIN
#define KS_NAMESPACE_END
#define USING_NAMESPACE_KS
#endif
// clang-format on

// Finally if no namespace, define this to empty (kinda stupid...)
#ifndef KS_NAMESPACE
#define KS_NAMESPACE
#endif