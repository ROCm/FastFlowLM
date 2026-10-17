/// \file audio_decoder.cpp
/// \brief Container/codec decoding for AudioReader, built on permissively
///        licensed decoders: dr_wav/dr_flac/dr_mp3 (public domain or MIT-0),
///        libvorbis/libogg and libopus (BSD-3-Clause).
/// \author FastFlowLM Team

#include "audio/audio_decoder.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>

#define DR_WAV_IMPLEMENTATION
#define DR_WAV_NO_STDIO
#include "dr_libs/dr_wav.h"
#define DR_FLAC_IMPLEMENTATION
#define DR_FLAC_NO_STDIO
#include "dr_libs/dr_flac.h"
#define DR_MP3_IMPLEMENTATION
#define DR_MP3_NO_STDIO
#include "dr_libs/dr_mp3.h"

#include <opus/opus_multistream.h>
#include <vorbis/vorbisfile.h>

namespace audio {
namespace {

constexpr size_t kChunkFrames = 65536;
constexpr int kMaxChannels = 64;

const char* const kUnsupportedHint =
    "supported formats are WAV, FLAC, MP3, Ogg (Vorbis/Opus/FLAC) and WebM/Matroska (Opus)";

uint16_t le16(const uint8_t* p) { return static_cast<uint16_t>(p[0] | (p[1] << 8)); }
uint32_t le32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}
uint64_t le64(const uint8_t* p) { return le32(p) | (static_cast<uint64_t>(le32(p + 4)) << 32); }

bool has_bytes(const uint8_t* data, size_t size, size_t offset, const char* magic, size_t len) {
    return offset + len <= size && std::memcmp(data + offset, magic, len) == 0;
}

bool valid_layout(int sample_rate, int channels) {
    return sample_rate > 0 && channels > 0 && channels <= kMaxChannels;
}

// ———————————————————————————————————————————————
// Format detection
// ———————————————————————————————————————————————
enum class Format { Wav, Flac, Mp3, Ogg, Matroska, Mp4, Adts, Unknown };

// Size of a leading ID3v2 tag (0 if none).
size_t id3v2_size(const uint8_t* data, size_t size) {
    if (size < 10 || !has_bytes(data, size, 0, "ID3", 3)) {
        return 0;
    }
    const size_t body = (static_cast<size_t>(data[6] & 0x7F) << 21) | (static_cast<size_t>(data[7] & 0x7F) << 14) |
                        (static_cast<size_t>(data[8] & 0x7F) << 7) | static_cast<size_t>(data[9] & 0x7F);
    const size_t footer = (data[5] & 0x10) ? 10 : 0;
    return std::min(size, 10 + body + footer);
}

Format sniff(const uint8_t* data, size_t size, size_t& offset) {
    offset = 0;
    static const uint8_t kW64Riff[16] = {'r', 'i', 'f', 'f', 0x2E, 0x91, 0xCF, 0x11,
                                         0xA5, 0xD6, 0x28, 0xDB, 0x04, 0xC1, 0x00, 0x00};
    if ((has_bytes(data, size, 0, "RIFF", 4) || has_bytes(data, size, 0, "RF64", 4) ||
         has_bytes(data, size, 0, "BW64", 4)) && has_bytes(data, size, 8, "WAVE", 4)) {
        return Format::Wav;
    }
    if (has_bytes(data, size, 0, "FORM", 4) &&
        (has_bytes(data, size, 8, "AIFF", 4) || has_bytes(data, size, 8, "AIFC", 4))) {
        return Format::Wav;
    }
    if (size >= 16 && std::memcmp(data, kW64Riff, 16) == 0) {
        return Format::Wav;
    }
    if (has_bytes(data, size, 0, "OggS", 4)) {
        return Format::Ogg;
    }
    if (has_bytes(data, size, 0, "\x1A\x45\xDF\xA3", 4)) {
        return Format::Matroska;
    }
    if (has_bytes(data, size, 4, "ftyp", 4)) {
        return Format::Mp4;
    }
    offset = id3v2_size(data, size);
    if (has_bytes(data, size, offset, "fLaC", 4)) {
        return Format::Flac;
    }
    if (offset + 2 <= size && data[offset] == 0xFF && (data[offset + 1] & 0xE0) == 0xE0) {
        // MPEG audio frame sync; layer bits 00 mean ADTS (AAC).
        return ((data[offset + 1] >> 1) & 0x03) == 0 ? Format::Adts : Format::Mp3;
    }
    if (offset > 0) {
        offset = 0;
        return Format::Mp3;  // ID3-tagged; let dr_mp3 find the first frame.
    }
    return Format::Unknown;
}

// ———————————————————————————————————————————————
// dr_libs / libvorbis
// ———————————————————————————————————————————————
bool decode_wav(const uint8_t* data, size_t size, DecodedAudio& out, std::string& error) {
    drwav wav;
    if (!drwav_init_memory(&wav, data, size, nullptr)) {
        error = "invalid or unsupported WAV/AIFF file";
        return false;
    }
    out.sample_rate = static_cast<int>(wav.sampleRate);
    out.channels = static_cast<int>(wav.channels);
    if (!valid_layout(out.sample_rate, out.channels)) {
        drwav_uninit(&wav);
        error = "WAV file has an invalid sample rate or channel count";
        return false;
    }
    // dr_wav maps 8-bit PCM as u8/255*2-1; FFmpeg (and libsndfile) use
    // (u8-128)/128, so convert those samples here to keep the previous values.
    const bool u8_pcm = wav.translatedFormatTag == DR_WAVE_FORMAT_PCM && wav.bitsPerSample == 8;
    std::vector<float> chunk(kChunkFrames * out.channels);
    std::vector<uint8_t> raw(u8_pcm ? kChunkFrames * out.channels : 0);
    for (;;) {
        const drwav_uint64 n = u8_pcm ? drwav_read_pcm_frames(&wav, kChunkFrames, raw.data())
                                      : drwav_read_pcm_frames_f32(&wav, kChunkFrames, chunk.data());
        if (n == 0) {
            break;
        }
        if (u8_pcm) {
            for (size_t i = 0; i < n * out.channels; ++i) {
                chunk[i] = (static_cast<float>(raw[i]) - 128.0f) / 128.0f;
            }
        }
        out.samples.insert(out.samples.end(), chunk.begin(), chunk.begin() + n * out.channels);
    }
    drwav_uninit(&wav);
    return true;
}

bool decode_flac(const uint8_t* data, size_t size, DecodedAudio& out, std::string& error) {
    drflac* flac = drflac_open_memory(data, size, nullptr);
    if (!flac) {
        error = "invalid FLAC file";
        return false;
    }
    out.sample_rate = static_cast<int>(flac->sampleRate);
    out.channels = static_cast<int>(flac->channels);
    if (!valid_layout(out.sample_rate, out.channels)) {
        drflac_close(flac);
        error = "FLAC file has an invalid sample rate or channel count";
        return false;
    }
    std::vector<float> chunk(kChunkFrames * out.channels);
    for (;;) {
        const drflac_uint64 n = drflac_read_pcm_frames_f32(flac, kChunkFrames, chunk.data());
        if (n == 0) {
            break;
        }
        out.samples.insert(out.samples.end(), chunk.begin(), chunk.begin() + n * out.channels);
    }
    drflac_close(flac);
    return true;
}

bool decode_mp3(const uint8_t* data, size_t size, DecodedAudio& out, std::string& error) {
    drmp3 mp3;
    if (!drmp3_init_memory(&mp3, data, size, nullptr)) {
        error = "invalid MP3 file";
        return false;
    }
    out.sample_rate = static_cast<int>(mp3.sampleRate);
    out.channels = static_cast<int>(mp3.channels);
    if (!valid_layout(out.sample_rate, out.channels)) {
        drmp3_uninit(&mp3);
        error = "MP3 file has an invalid sample rate or channel count";
        return false;
    }
    std::vector<float> chunk(kChunkFrames * out.channels);
    for (;;) {
        const drmp3_uint64 n = drmp3_read_pcm_frames_f32(&mp3, kChunkFrames, chunk.data());
        if (n == 0) {
            break;
        }
        out.samples.insert(out.samples.end(), chunk.begin(), chunk.begin() + n * out.channels);
    }
    drmp3_uninit(&mp3);
    return true;
}

// vorbisfile reads through callbacks; serve them from the in-memory file.
struct MemoryFile {
    const uint8_t* data;
    size_t size;
    size_t pos;
};

size_t memory_read(void* ptr, size_t size, size_t nmemb, void* source) {
    auto* f = static_cast<MemoryFile*>(source);
    if (size == 0) {
        return 0;
    }
    const size_t n = std::min(nmemb, (f->size - f->pos) / size);
    std::memcpy(ptr, f->data + f->pos, n * size);
    f->pos += n * size;
    return n;
}

int memory_seek(void* source, ogg_int64_t offset, int whence) {
    auto* f = static_cast<MemoryFile*>(source);
    int64_t base = 0;
    if (whence == SEEK_CUR) {
        base = static_cast<int64_t>(f->pos);
    } else if (whence == SEEK_END) {
        base = static_cast<int64_t>(f->size);
    } else if (whence != SEEK_SET) {
        return -1;
    }
    const int64_t target = base + offset;
    if (target < 0 || target > static_cast<int64_t>(f->size)) {
        return -1;
    }
    f->pos = static_cast<size_t>(target);
    return 0;
}

long memory_tell(void* source) { return static_cast<long>(static_cast<MemoryFile*>(source)->pos); }

bool decode_vorbis(const uint8_t* data, size_t size, DecodedAudio& out, std::string& error) {
    MemoryFile file{data, size, 0};
    const ov_callbacks callbacks{memory_read, memory_seek, nullptr, memory_tell};
    OggVorbis_File vf;
    if (ov_open_callbacks(&file, &vf, nullptr, 0, callbacks) < 0) {
        error = "invalid Ogg Vorbis file";
        return false;
    }
    const vorbis_info* info = ov_info(&vf, -1);
    out.sample_rate = info ? static_cast<int>(info->rate) : 0;
    out.channels = info ? info->channels : 0;
    if (!valid_layout(out.sample_rate, out.channels)) {
        ov_clear(&vf);
        error = "Ogg Vorbis file has an invalid sample rate or channel count";
        return false;
    }
    int first_section = -1;
    for (;;) {
        float** pcm = nullptr;
        int section = 0;
        const long n = ov_read_float(&vf, &pcm, 4096, &section);
        if (n == OV_HOLE) {
            continue;  // damaged data; keep going, as FFmpeg does
        }
        if (n <= 0) {
            break;
        }
        if (first_section < 0) {
            first_section = section;
        } else if (section != first_section) {
            break;  // chained stream: decode the first link only
        }
        const size_t base = out.samples.size();
        out.samples.resize(base + static_cast<size_t>(n) * out.channels);
        for (long i = 0; i < n; ++i) {
            for (int c = 0; c < out.channels; ++c) {
                out.samples[base + static_cast<size_t>(i) * out.channels + c] = pcm[c][i];
            }
        }
    }
    ov_clear(&vf);
    return true;
}

// ———————————————————————————————————————————————
// Opus (RFC 7845 header; packets from Ogg or Matroska)
// ———————————————————————————————————————————————
struct OpusPacket {
    std::vector<uint8_t> data;
    int64_t discard_end = 0;  // samples (48 kHz) to drop from the end of this packet's output
};

bool decode_opus(const std::vector<uint8_t>& head, const std::vector<OpusPacket>& packets,
                 int64_t total_samples, DecodedAudio& out, std::string& error) {
    if (head.size() < 19 || std::memcmp(head.data(), "OpusHead", 8) != 0 || (head[8] >> 4) != 0) {
        error = "invalid Opus header";
        return false;
    }
    const int channels = head[9];
    const int pre_skip = le16(&head[10]);
    const int16_t gain_q8 = static_cast<int16_t>(le16(&head[16]));
    const int family = head[18];

    int streams = 1;
    int coupled = channels > 1 ? 1 : 0;
    unsigned char mapping[255] = {0, 1};
    if (family == 0) {
        if (channels < 1 || channels > 2) {
            error = "invalid Opus channel count";
            return false;
        }
    } else if (family == 1 || family == 255) {
        if (channels < 1 || head.size() < 21u + channels) {
            error = "invalid Opus channel mapping";
            return false;
        }
        streams = head[19];
        coupled = head[20];
        std::memcpy(mapping, &head[21], channels);
    } else {
        error = "unsupported Opus channel mapping family " + std::to_string(family);
        return false;
    }

    int err = 0;
    std::unique_ptr<OpusMSDecoder, decltype(&opus_multistream_decoder_destroy)> dec(
        opus_multistream_decoder_create(48000, channels, streams, coupled, mapping, &err),
        opus_multistream_decoder_destroy);
    if (!dec || err != OPUS_OK) {
        error = std::string("could not create Opus decoder: ") + opus_strerror(err);
        return false;
    }

    constexpr int kMaxFrame = 5760;  // 120 ms at 48 kHz
    std::vector<float> pcm(static_cast<size_t>(kMaxFrame) * channels);
    for (const OpusPacket& p : packets) {
        if (p.data.empty() || p.data.size() > static_cast<size_t>(INT32_MAX)) {
            continue;
        }
        int n = opus_multistream_decode_float(dec.get(), p.data.data(), static_cast<opus_int32>(p.data.size()),
                                              pcm.data(), kMaxFrame, 0);
        if (n < 0) {
            continue;  // skip a corrupt packet, as FFmpeg does
        }
        n -= static_cast<int>(std::min<int64_t>(std::max<int64_t>(p.discard_end, 0), n));
        out.samples.insert(out.samples.end(), pcm.begin(), pcm.begin() + static_cast<size_t>(n) * channels);
    }

    const size_t skip = std::min(out.samples.size(), static_cast<size_t>(pre_skip) * channels);
    out.samples.erase(out.samples.begin(), out.samples.begin() + skip);
    if (total_samples >= 0 && static_cast<size_t>(total_samples) * channels < out.samples.size()) {
        out.samples.resize(static_cast<size_t>(total_samples) * channels);
    }
    if (gain_q8 != 0) {
        const float gain = static_cast<float>(std::pow(10.0, gain_q8 / (20.0 * 256.0)));
        for (float& s : out.samples) {
            s *= gain;
        }
    }
    out.sample_rate = 48000;
    out.channels = channels;
    return true;
}

// ———————————————————————————————————————————————
// Ogg (RFC 3533): packets of the first logical stream
// ———————————————————————————————————————————————
struct OggStream {
    std::vector<std::vector<uint8_t>> packets;
    std::vector<size_t> packet_page;    // page on which each packet completes
    std::vector<int64_t> page_granule;  // granule position of each page of the stream
    int64_t last_granule = -1;
};

bool read_ogg_stream(const uint8_t* data, size_t size, OggStream& stream) {
    int64_t serial = -1;
    std::vector<uint8_t> partial;
    size_t pos = 0;
    while (pos + 27 <= size) {
        if (std::memcmp(data + pos, "OggS", 4) != 0 || data[pos + 4] != 0) {
            break;
        }
        const uint8_t header_type = data[pos + 5];
        const int64_t granule = static_cast<int64_t>(le64(data + pos + 6));
        const uint32_t page_serial = le32(data + pos + 14);
        const size_t segments = data[pos + 26];
        const size_t body = pos + 27 + segments;
        if (body > size) {
            break;
        }
        size_t body_len = 0;
        for (size_t i = 0; i < segments; ++i) {
            body_len += data[pos + 27 + i];
        }
        if (body + body_len > size) {
            break;
        }
        if (serial < 0 && (header_type & 0x02)) {
            serial = page_serial;
        }
        if (static_cast<int64_t>(page_serial) == serial) {
            const size_t page = stream.page_granule.size();
            stream.page_granule.push_back(granule);
            size_t off = body;
            for (size_t i = 0; i < segments; ++i) {
                const size_t len = data[pos + 27 + i];
                partial.insert(partial.end(), data + off, data + off + len);
                off += len;
                if (len < 255) {
                    stream.packets.push_back(std::move(partial));
                    stream.packet_page.push_back(page);
                    partial.clear();
                }
            }
            if (granule != -1) {
                stream.last_granule = granule;
            }
        }
        pos = body + body_len;
    }
    return !stream.packets.empty();
}

bool decode_ogg(const uint8_t* data, size_t size, DecodedAudio& out, std::string& error) {
    OggStream stream;
    if (!read_ogg_stream(data, size, stream)) {
        error = "invalid Ogg file";
        return false;
    }
    const std::vector<uint8_t>& first = stream.packets.front();
    if (first.size() >= 7 && std::memcmp(first.data(), "\x01vorbis", 7) == 0) {
        return decode_vorbis(data, size, out, error);
    }
    if (first.size() >= 5 && std::memcmp(first.data(), "\x7F""FLAC", 5) == 0) {
        return decode_flac(data, size, out, error);
    }
    if (first.size() >= 8 && std::memcmp(first.data(), "OpusHead", 8) == 0) {
        // Packet 0 is OpusHead, packet 1 OpusTags; audio follows. Playback
        // length (RFC 7845, 4.3-4.5): last granule minus the stream's start
        // position minus pre-skip, where the start position is the first audio
        // page's granule minus the samples of the packets completing on it.
        const int pre_skip = first.size() >= 12 ? le16(&first[10]) : 0;
        int64_t start = 0;
        if (stream.packets.size() > 2) {
            const size_t first_page = stream.packet_page[2];
            int64_t first_samples = 0;
            for (size_t i = 2; i < stream.packets.size() && stream.packet_page[i] == first_page; ++i) {
                const std::vector<uint8_t>& pkt = stream.packets[i];
                const int n = pkt.empty() ? 0 : opus_packet_get_nb_samples(pkt.data(), static_cast<opus_int32>(pkt.size()), 48000);
                first_samples += std::max(n, 0);
            }
            const int64_t granule = stream.page_granule[first_page];
            start = granule >= 0 ? std::max<int64_t>(granule - first_samples, 0) : 0;
        }
        const int64_t total = stream.last_granule >= 0 ? stream.last_granule - start - pre_skip : -1;

        std::vector<OpusPacket> packets;
        for (size_t i = 2; i < stream.packets.size(); ++i) {
            packets.push_back({std::move(stream.packets[i]), 0});
        }
        return decode_opus(first, packets, total >= 0 ? total : -1, out, error);
    }
    error = std::string("unsupported codec in Ogg file; ") + kUnsupportedHint;
    return false;
}

// ———————————————————————————————————————————————
// Matroska / WebM: Opus audio track only
// ———————————————————————————————————————————————
namespace ebml {
constexpr uint32_t kSegment = 0x18538067, kTracks = 0x1654AE6B, kTrackEntry = 0xAE, kTrackNumber = 0xD7,
                   kTrackType = 0x83, kCodecId = 0x86, kCodecPrivate = 0x63A2, kCluster = 0x1F43B675,
                   kSimpleBlock = 0xA3, kBlockGroup = 0xA0, kBlock = 0xA1, kDiscardPadding = 0x75A2;

// Element ID: 1-4 bytes, length marker kept.
bool read_id(const uint8_t* d, size_t n, size_t& pos, uint32_t& id) {
    if (pos >= n || d[pos] == 0) {
        return false;
    }
    int len = 1;
    while (len <= 4 && !(d[pos] & (0x80 >> (len - 1)))) {
        ++len;
    }
    if (len > 4 || pos + len > n) {
        return false;
    }
    id = 0;
    for (int i = 0; i < len; ++i) {
        id = (id << 8) | d[pos + i];
    }
    pos += len;
    return true;
}

// Variable-size integer: 1-8 bytes, length marker removed. `unknown` is set
// when all value bits are 1 (unknown element size).
bool read_vint(const uint8_t* d, size_t n, size_t& pos, uint64_t& value, bool* unknown = nullptr, int* length = nullptr) {
    if (pos >= n || d[pos] == 0) {
        return false;
    }
    int len = 1;
    while (!(d[pos] & (0x80 >> (len - 1)))) {
        ++len;
    }
    if (pos + len > n) {
        return false;
    }
    value = d[pos] & (0xFF >> len);
    bool all_ones = value == (0xFFu >> len);
    for (int i = 1; i < len; ++i) {
        value = (value << 8) | d[pos + i];
        all_ones = all_ones && d[pos + i] == 0xFF;
    }
    if (unknown) {
        *unknown = all_ones;
    }
    if (length) {
        *length = len;
    }
    pos += len;
    return true;
}

uint64_t read_uint(const uint8_t* d, size_t len) {
    uint64_t v = 0;
    for (size_t i = 0; i < len && i < 8; ++i) {
        v = (v << 8) | d[i];
    }
    return v;
}

int64_t read_sint(const uint8_t* d, size_t len) {
    if (len == 0 || len > 8) {
        return 0;
    }
    uint64_t v = read_uint(d, len);
    if (d[0] & 0x80) {
        v |= len == 8 ? 0 : (~uint64_t{0} << (8 * len));
    }
    return static_cast<int64_t>(v);
}

// Split a (Simple)Block payload into frames for `track` (RFC 9559, 10.2).
bool block_frames(const uint8_t* d, size_t n, uint64_t track, std::vector<std::vector<uint8_t>>& frames) {
    size_t pos = 0;
    uint64_t number = 0;
    if (!read_vint(d, n, pos, number) || pos + 3 > n) {
        return false;
    }
    if (number != track) {
        return true;
    }
    const uint8_t flags = d[pos + 2];
    pos += 3;
    const int lacing = (flags >> 1) & 0x03;
    if (lacing == 0) {
        frames.emplace_back(d + pos, d + n);
        return true;
    }
    if (pos >= n) {
        return false;
    }
    const size_t count = static_cast<size_t>(d[pos++]) + 1;
    std::vector<uint64_t> sizes;
    if (lacing == 1) {  // Xiph
        for (size_t i = 0; i + 1 < count; ++i) {
            uint64_t s = 0;
            uint8_t b = 0;
            do {
                if (pos >= n) {
                    return false;
                }
                b = d[pos++];
                s += b;
            } while (b == 255);
            sizes.push_back(s);
        }
    } else if (lacing == 3) {  // EBML
        uint64_t s = 0;
        if (count > 1) {
            if (!read_vint(d, n, pos, s)) {
                return false;
            }
            sizes.push_back(s);
        }
        for (size_t i = 1; i + 1 < count; ++i) {
            uint64_t raw = 0;
            int len = 0;
            if (!read_vint(d, n, pos, raw, nullptr, &len)) {
                return false;
            }
            const int64_t delta = static_cast<int64_t>(raw) - ((int64_t{1} << (7 * len - 1)) - 1);
            const int64_t next = static_cast<int64_t>(s) + delta;
            if (next < 0) {
                return false;
            }
            s = static_cast<uint64_t>(next);
            sizes.push_back(s);
        }
    } else {  // fixed
        const size_t each = (n - pos) / count;
        if (each * count != n - pos) {
            return false;
        }
        sizes.assign(count - 1, each);
    }
    for (uint64_t s : sizes) {
        if (s > n - pos) {
            return false;
        }
        frames.emplace_back(d + pos, d + pos + s);
        pos += static_cast<size_t>(s);
    }
    frames.emplace_back(d + pos, d + n);
    return true;
}
} // namespace ebml

bool decode_matroska(const uint8_t* data, size_t size, DecodedAudio& out, std::string& error) {
    struct Track {
        uint64_t number = 0;
        uint64_t type = 0;
        std::string codec;
        std::vector<uint8_t> codec_private;
    };
    std::vector<Track> tracks;
    int audio_index = -1;  // index into tracks once the first block selects it
    std::vector<OpusPacket> packets;
    size_t group_first_packet = 0;
    int64_t pending_discard_ns = 0;

    // Elements are visited in file order. Master elements we need children
    // from are entered instead of skipped, which also copes with the
    // unknown-size Segment/Cluster elements live recorders (MediaRecorder)
    // write.
    size_t pos = 0;
    while (pos < size) {
        uint32_t id = 0;
        uint64_t len = 0;
        bool unknown = false;
        if (!ebml::read_id(data, size, pos, id) || !ebml::read_vint(data, size, pos, len, &unknown)) {
            break;
        }
        if (id == ebml::kSegment || id == ebml::kTracks || id == ebml::kCluster) {
            continue;
        }
        if (id == ebml::kTrackEntry) {
            if (audio_index < 0) {
                tracks.emplace_back();
            }
            continue;
        }
        if (id == ebml::kBlockGroup) {
            group_first_packet = packets.size();
            pending_discard_ns = 0;
            continue;
        }
        if (unknown) {
            break;  // only master elements may have an unknown size
        }
        if (len > size - pos) {
            len = size - pos;  // truncated file: use what is there
        }
        const uint8_t* body = data + pos;
        const size_t blen = static_cast<size_t>(len);
        pos += blen;

        if (!tracks.empty() && audio_index < 0) {
            Track& t = tracks.back();
            if (id == ebml::kTrackNumber) {
                t.number = ebml::read_uint(body, blen);
            } else if (id == ebml::kTrackType) {
                t.type = ebml::read_uint(body, blen);
            } else if (id == ebml::kCodecId) {
                // Codec IDs are short ASCII strings; keep untrusted bytes out of
                // error messages.
                t.codec.clear();
                for (size_t i = 0; i < blen && t.codec.size() < 32; ++i) {
                    const char ch = static_cast<char>(body[i]);
                    if (ch == '\0') {
                        break;
                    }
                    t.codec.push_back(ch >= 0x20 && ch < 0x7F ? ch : '?');
                }
            } else if (id == ebml::kCodecPrivate) {
                t.codec_private.assign(body, body + blen);
            }
        }
        if (id == ebml::kSimpleBlock || id == ebml::kBlock) {
            if (audio_index < 0) {
                for (size_t i = 0; i < tracks.size(); ++i) {
                    if (tracks[i].type == 2) {
                        audio_index = static_cast<int>(i);
                        break;
                    }
                }
                if (audio_index < 0) {
                    error = "no audio track in Matroska/WebM file";
                    return false;
                }
                if (tracks[audio_index].codec != "A_OPUS") {
                    error = "unsupported audio codec " + tracks[audio_index].codec + " in Matroska/WebM file; " +
                            kUnsupportedHint;
                    return false;
                }
            }
            std::vector<std::vector<uint8_t>> frames;
            if (!ebml::block_frames(body, blen, tracks[audio_index].number, frames)) {
                continue;
            }
            for (auto& f : frames) {
                packets.push_back({std::move(f), 0});
            }
            if (id == ebml::kBlock && pending_discard_ns > 0 && packets.size() > group_first_packet) {
                packets.back().discard_end = pending_discard_ns * 48000 / 1000000000;
                pending_discard_ns = 0;
            }
        } else if (id == ebml::kDiscardPadding) {
            const int64_t ns = ebml::read_sint(body, blen);
            if (packets.size() > group_first_packet) {
                packets.back().discard_end = ns * 48000 / 1000000000;
            } else {
                pending_discard_ns = ns;
            }
        }
    }

    if (audio_index < 0) {
        error = "no audio data in Matroska/WebM file";
        return false;
    }
    return decode_opus(tracks[audio_index].codec_private, packets, -1, out, error);
}

} // namespace

bool decode_audio_file(const uint8_t* data, size_t size, DecodedAudio& out, std::string& error) {
    out = DecodedAudio{};
    if (!data || size == 0) {
        error = "empty audio data";
        return false;
    }
    size_t offset = 0;
    bool ok = false;
    switch (sniff(data, size, offset)) {
        case Format::Wav:
            ok = decode_wav(data, size, out, error);
            break;
        case Format::Flac:
            ok = decode_flac(data + offset, size - offset, out, error);
            break;
        case Format::Mp3:
            ok = decode_mp3(data, size, out, error);
            break;
        case Format::Ogg:
            ok = decode_ogg(data, size, out, error);
            break;
        case Format::Matroska:
            ok = decode_matroska(data, size, out, error);
            break;
        case Format::Mp4:
        case Format::Adts:
            error = std::string("AAC/M4A/MP4 audio is not supported; ") + kUnsupportedHint;
            return false;
        case Format::Unknown:
            error = std::string("unrecognized audio format; ") + kUnsupportedHint;
            return false;
    }
    if (ok && out.samples.empty()) {
        error = "audio file contains no samples";
        return false;
    }
    return ok;
}

} // namespace audio
