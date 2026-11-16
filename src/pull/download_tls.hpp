/// \file download_tls.hpp
/// \brief how model downloads verify TLS
/// \note Every transfer verifies the server's certificate and host name, libcurl's own default.
///       On Windows libcurl uses the system certificate store (the shipped libcurl is
///       Schannel; CURLSSLOPT_NATIVE_CA does the same for an OpenSSL build), so a corporate
///       root CA installed in Windows is trusted. A revocation check that cannot be completed
///       (an offline CRL or OCSP server behind a proxy) does not fail the download
///       (CURLSSLOPT_REVOKE_BEST_EFFORT); a revoked certificate still does.
///       FLM_INSECURE_DOWNLOAD=1 turns verification off, explicitly and with a warning.
///       No curl header here, so a host test can check the settings without libcurl.
#pragma once

#include <cstdlib>
#include <cstring>
#include <string>

namespace download_utils {

/// what download_model.cpp passes to CURLOPT_SSL_VERIFYPEER / CURLOPT_SSL_VERIFYHOST / CURLOPT_SSL_OPTIONS
struct TlsSettings {
    long verify_peer = 1;
    long verify_host = 2;
    bool native_ca = false;              ///< CURLSSLOPT_NATIVE_CA (Windows)
    bool revoke_best_effort = false;     ///< CURLSSLOPT_REVOKE_BEST_EFFORT (Windows)
};

/// FLM_INSECURE_DOWNLOAD set to anything but "" or "0"
inline bool insecure_download_requested() {
    const char* value = std::getenv("FLM_INSECURE_DOWNLOAD");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

/// verification on, whatever FLM_INSECURE_DOWNLOAD says: for a request that carries a credential
inline TlsSettings verified_tls_settings() {
    TlsSettings s;
#ifdef _WIN32
    s.native_ca = true;
    s.revoke_best_effort = true;
#endif
    return s;
}

/// the TLS settings for a download: verified, unless FLM_INSECURE_DOWNLOAD asks otherwise
inline TlsSettings tls_settings() {
    if (!insecure_download_requested()) return verified_tls_settings();
    TlsSettings s;
    s.verify_peer = 0;
    s.verify_host = 0;
    return s;
}

/// whether a curl result code means the certificate (or the TLS handshake) was refused:
/// CURLE_SSL_CONNECT_ERROR (35), CURLE_PEER_FAILED_VERIFICATION (60), CURLE_SSL_CACERT_BADFILE (77),
/// CURLE_SSL_ISSUER_ERROR (83)
inline bool is_tls_error(int curl_code) {
    return curl_code == 35 || curl_code == 60 || curl_code == 77 || curl_code == 83;
}

/// the line printed after a TLS failure: what to check, and the way out
inline std::string tls_failure_hint() {
    return "The server's TLS certificate could not be verified. Behind a corporate proxy, install its root "
           "certificate in the system store; to download without verification (insecure), set "
           "FLM_INSECURE_DOWNLOAD=1. Requests that carry a Hugging Face token are always verified: for those, "
           "only the root certificate helps.";
}

}  // namespace download_utils
