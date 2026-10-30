#!/usr/bin/env bash
# Fail if a Linux package ships, links or depends on a library FLM dropped for
# licensing reasons: FFmpeg and the GPL codecs distro FFmpeg pulls in (x264,
# x265, xvid, libdvdnav/libdvdread), FFTW and GNU readline. Also fail if it
# bundles an (L)GPL system library: the ones distro FFmpeg and libcurl pulled
# into the HRX tarball, glibc pieces and the GCC runtime. Packages may link
# those, but they come from the host.
#
# Usage: check_no_copyleft_libs.sh <package.tar.gz | package.deb>...
set -euo pipefail

# Shared-library file names (bundled files and ELF NEEDED entries).
lib_re='^lib(av(codec|format|util|filter|device)|sw(scale|resample)|postproc|x264|x265|xvidcore|dvdnav|dvdread|fftw3[a-z_]*|readline)[.]so'
# Shared-library file names that must not be bundled (linking is fine).
bundle_re='^lib(codec2|gme|gdk_pixbuf-2[.]0|(glib|gio|gobject|gmodule|gthread)-2[.]0|fribidi|idn2|tasn1|unistring|gmp|gnutls|keyutils|mp3lame|bluray|datrie|numa|rsvg-2|rtmp|soxr|ssh|thai|udfread|mpg123|nettle|hogweed|pgm-5[.]3|pango(cairo|ft2)?-1[.]0|shine|twolame|blkid|mount|zvbi|mvec|resolv|stdc[+][+]|gcc_s|gomp|atomic)[.]so'
# Debian binary package names (.deb Depends).
pkg_re='^(libav(codec|format|util|filter|device)|libsw(scale|resample)|libpostproc|libx26[45]|libxvidcore|libdvd(nav|read)|libfftw3|libreadline)'

status=0
for pkg in "$@"; do
    tmp=$(mktemp -d)
    case "$pkg" in
        *.deb)
            if dpkg-deb -f "$pkg" Depends | tr ',|' '\n\n' | sed 's/^ *//' | grep -E "$pkg_re"; then
                echo "::error::$pkg depends on a package listed above"
                status=1
            fi
            dpkg-deb -x "$pkg" "$tmp" ;;
        *.tar.gz)
            tar xzf "$pkg" -C "$tmp" ;;
        *)
            echo "::error::unknown package type: $pkg"
            exit 2 ;;
    esac

    if find "$tmp" \( -type f -o -type l \) -printf '%f\n' | grep -E "$lib_re"; then
        echo "::error::$pkg bundles a library listed above"
        status=1
    fi
    if find "$tmp" \( -type f -o -type l \) -printf '%f\n' | grep -E "$bundle_re"; then
        echo "::error::$pkg bundles an (L)GPL system library listed above; it must come from the host"
        status=1
    fi
    while IFS= read -r -d '' f; do
        if readelf -d "$f" 2>/dev/null | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p' | grep -E "$lib_re"; then
            echo "::error::${f#"$tmp"/} in $pkg links a library listed above"
            status=1
        fi
    done < <(find "$tmp" -type f -print0)
    rm -rf "$tmp"
    [ "$status" -eq 0 ] && echo "$pkg: no FFmpeg, GPL codec, FFTW or readline libraries, no bundled (L)GPL system libraries"
done
exit "$status"
