/* <editor-fold desc="MIT License">

Copyright(c) 2025 Robert Osfield

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

</editor-fold> */

#include <vsgXchange/3DTiles.h>

#include <vsg/io/Path.h>
#include <vsg/io/mem_stream.h>
#include <vsg/io/read.h>
#include <vsg/nodes/MatrixTransform.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

using namespace vsgXchange;

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// pnts_FeatureTable
//
void Tiles3D::pnts_FeatureTable::read_array(vsg::JSONParser& parser, const std::string_view& property)
{
    if (property == "POSITION") parser.read_array(POSITION);
    else if (property == "POSITION_QUANTIZED") parser.read_array(POSITION_QUANTIZED);
    else if (property == "RGBA") parser.read_array(RGBA);
    else if (property == "RGB") parser.read_array(RGB);
    else if (property == "RGB565") parser.read_array(RGB565);
    else if (property == "CONSTANT_RGBA") parser.read_array(CONSTANT_RGBA);
    else if (property == "NORMAL") parser.read_array(NORMAL);
    else if (property == "NORMAL_OCT16P") parser.read_array(NORMAL_OCT16P);
    else if (property == "RTC_CENTER") parser.read_array(RTC_CENTER);
    else if (property == "QUANTIZED_VOLUME_OFFSET") parser.read_array(QUANTIZED_VOLUME_OFFSET);
    else if (property == "QUANTIZED_VOLUME_SCALE") parser.read_array(QUANTIZED_VOLUME_SCALE);
    else parser.warning();
}

void Tiles3D::pnts_FeatureTable::read_object(vsg::JSONParser& parser, const std::string_view& property)
{
    if (property == "POSITION") parser.read_object(POSITION);
    else if (property == "POSITION_QUANTIZED") parser.read_object(POSITION_QUANTIZED);
    else if (property == "RGBA") parser.read_object(RGBA);
    else if (property == "RGB") parser.read_object(RGB);
    else if (property == "RGB565") parser.read_object(RGB565);
    else if (property == "CONSTANT_RGBA") parser.read_object(CONSTANT_RGBA);
    else if (property == "NORMAL") parser.read_object(NORMAL);
    else if (property == "NORMAL_OCT16P") parser.read_object(NORMAL_OCT16P);
    else if (property == "RTC_CENTER") parser.read_object(RTC_CENTER);
    else if (property == "QUANTIZED_VOLUME_OFFSET") parser.read_object(QUANTIZED_VOLUME_OFFSET);
    else if (property == "QUANTIZED_VOLUME_SCALE") parser.read_object(QUANTIZED_VOLUME_SCALE);
    else parser.warning();
}

void Tiles3D::pnts_FeatureTable::read_number(vsg::JSONParser& parser, const std::string_view& property, std::istream& input)
{
    if (property == "POINTS_LENGTH") input >> POINTS_LENGTH;
    else if (property == "BATCH_LENGTH") input >> BATCH_LENGTH;
    else parser.warning();
}

void Tiles3D::pnts_FeatureTable::convert()
{
    if (POINTS_LENGTH == 0 || !binary) return;

    const uint32_t N = POINTS_LENGTH;
    POSITION.assign(*binary, 3 * N);
    POSITION_QUANTIZED.assign(*binary, 3 * N);
    RGBA.assign(*binary, 4 * N);
    RGB.assign(*binary, 3 * N);
    RGB565.assign(*binary, N);
    NORMAL.assign(*binary, 3 * N);
    NORMAL_OCT16P.assign(*binary, 2 * N);
    // Globals: fixed length regardless of the point count.
    CONSTANT_RGBA.assign(*binary, 4);
    RTC_CENTER.assign(*binary, 3);
    QUANTIZED_VOLUME_OFFSET.assign(*binary, 3);
    QUANTIZED_VOLUME_SCALE.assign(*binary, 3);
}

void Tiles3D::pnts_FeatureTable::report(vsg::LogOutput& output)
{
    output("pnts_FeatureTable { ");
    output("    POINTS_LENGTH ", POINTS_LENGTH);
    if (POSITION) output("    POSITION ", POSITION.values);
    if (POSITION_QUANTIZED) output("    POSITION_QUANTIZED ", POSITION_QUANTIZED.values);
    if (RGBA) output("    RGBA ", RGBA.values);
    if (RGB) output("    RGB ", RGB.values);
    if (RGB565) output("    RGB565 ", RGB565.values);
    if (CONSTANT_RGBA) output("    CONSTANT_RGBA ", CONSTANT_RGBA.values);
    if (NORMAL) output("    NORMAL ", NORMAL.values);
    if (RTC_CENTER) output("    RTC_CENTER ", RTC_CENTER.values);
    output("}");
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// read_pnts
//
namespace
{
    void append_u32(std::string& s, uint32_t v)
    {
        char b[4];
        std::memcpy(b, &v, 4);
        s.append(b, 4);
    }

    /// Wrap a POSITION/COLOR_0 buffer as a glTF binary with primitive mode 0.
    ///
    /// Delegating to the glTF reader rather than assembling a vsg::Geometry and
    /// a ShaderSet by hand is deliberate: mode 0 already maps to
    /// VK_PRIMITIVE_TOPOLOGY_POINT_LIST there, COLOR_0 already maps to
    /// vsg_Color, and the material, sRGB and instancing paths are the ones the
    /// rest of the 3D Tiles content already goes through. b3dm and i3dm both
    /// hand their payload to the same reader for the same reason.
    std::string buildPointsGlb(const std::vector<float>& positions,      // 3 per point
                               const std::vector<uint8_t>& colours,      // 4 per point, normalized
                               const vsg::vec3& bbMin, const vsg::vec3& bbMax)
    {
        const uint32_t count = static_cast<uint32_t>(positions.size() / 3);
        const uint32_t posBytes = count * 12;


        // COLOR_0 as FLOAT VEC4, not normalized UNSIGNED_BYTE. The compact
        // form is legal glTF and the reader accepts it, but the resulting
        // ubvec4 vertex buffer rendered every point pure black -- measured,
        // 20,469 of 20,469 non-background pixels exactly (0,0,0), against a
        // colour array that held the file's real values. Floats cost 16 bytes
        // a point instead of 4 and leave nothing to interpret.
        std::vector<float> fcolours;
        fcolours.reserve(colours.size());
        for (uint8_t c : colours) fcolours.push_back(static_cast<float>(c) / 255.0f);
        const uint32_t colBytesF = count * 16;

        std::string bin;
        bin.reserve(posBytes + colBytesF + 4);
        bin.append(reinterpret_cast<const char*>(positions.data()), posBytes);
        bin.append(reinterpret_cast<const char*>(fcolours.data()), colBytesF);
        while (bin.size() % 4 != 0) bin.push_back('\0');   // chunks are 4-byte aligned

        // POSITION accessors REQUIRE min/max; a reader is entitled to reject the
        // asset without them, and vsgXchange uses them for the bound.
        char json[1600];
        std::snprintf(json, sizeof(json),
            "{\"asset\":{\"version\":\"2.0\"},\"scene\":0,"
            "\"extensionsUsed\":[\"KHR_materials_unlit\"],"
            "\"scenes\":[{\"nodes\":[0]}],"
            "\"nodes\":[{\"mesh\":0,\"name\":\"points\"}],"
            "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0,\"COLOR_0\":1},"
            "\"mode\":0,\"material\":0}]}],"
            // KHR_materials_unlit, because a point has no surface and therefore
            // no meaningful normal. Lit, the shader takes whatever the default
            // normal is and shades every point the same way: the 125,000-point
            // Cesium sample rendered as a solid BLACK sphere while its colour
            // array held the file's real values. Unlit passes COLOR_0 straight
            // through, which is what a point cloud means.
            "\"materials\":[{\"pbrMetallicRoughness\":{\"baseColorFactor\":[1,1,1,1],"
            "\"metallicFactor\":0,\"roughnessFactor\":1},\"doubleSided\":true,"
            "\"extensions\":{\"KHR_materials_unlit\":{}}}],"
            "\"accessors\":["
              "{\"bufferView\":0,\"componentType\":5126,\"count\":%u,\"type\":\"VEC3\","
               "\"min\":[%.9g,%.9g,%.9g],\"max\":[%.9g,%.9g,%.9g]},"
              "{\"bufferView\":1,\"componentType\":5126,"
               "\"count\":%u,\"type\":\"VEC4\"}],"
            "\"bufferViews\":["
              "{\"buffer\":0,\"byteOffset\":0,\"byteLength\":%u,\"target\":34962},"
              "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34962}],"
            "\"buffers\":[{\"byteLength\":%u}]}",
            count, bbMin.x, bbMin.y, bbMin.z, bbMax.x, bbMax.y, bbMax.z,
            count, posBytes, posBytes, colBytesF,
            static_cast<uint32_t>(bin.size()));

        std::string js(json);
        while (js.size() % 4 != 0) js.push_back(' ');

        std::string glb;
        glb.append("glTF", 4);
        append_u32(glb, 2);
        append_u32(glb, static_cast<uint32_t>(12 + 8 + js.size() + 8 + bin.size()));
        append_u32(glb, static_cast<uint32_t>(js.size()));
        append_u32(glb, 0x4E4F534A);   // "JSON"
        glb += js;
        append_u32(glb, static_cast<uint32_t>(bin.size()));
        append_u32(glb, 0x004E4942);   // "BIN"
        glb += bin;
        return glb;
    }
}

vsg::ref_ptr<vsg::Object> Tiles3D::read_pnts(std::istream& fin, vsg::ref_ptr<const vsg::Options> options, const vsg::Path& filename) const
{
    fin.seekg(0);

    // https://github.com/CesiumGS/3d-tiles/blob/1.0/specification/TileFormats/PointCloud
    struct Header
    {
        char magic[4] = {0, 0, 0, 0};
        uint32_t version = 0;
        uint32_t byteLength = 0;
        uint32_t featureTableJSONByteLength = 0;
        uint32_t featureTableBinaryByteLength = 0;
        uint32_t batchTableJSONByteLength = 0;
        uint32_t batchTableBinaryLength = 0;
    } header;

    fin.read(reinterpret_cast<char*>(&header), sizeof(Header));
    if (!fin.good() || std::strncmp(header.magic, "pnts", 4) != 0) return {};

    vsg::JSONParser parser;
    parser.options = options;

    auto featureTable = pnts_FeatureTable::create();
    if (header.featureTableJSONByteLength > 0)
    {
        parser.buffer.resize(header.featureTableJSONByteLength);
        fin.read(parser.buffer.data(), header.featureTableJSONByteLength);

        if (header.featureTableBinaryByteLength > 0)
        {
            featureTable->binary = vsg::ubyteArray::create(header.featureTableBinaryByteLength);
            fin.read(reinterpret_cast<char*>(featureTable->binary->dataPointer()), header.featureTableBinaryByteLength);
        }

        parser.read_object(*featureTable);
        featureTable->convert();
    }

    // The batch table is read past but not yet applied; per-point metadata has
    // nowhere to go until the glTF reader carries feature IDs through.
    if (header.batchTableJSONByteLength > 0)
    {
        std::string skip;
        skip.resize(header.batchTableJSONByteLength + header.batchTableBinaryLength);
        fin.read(skip.data(), skip.size());
    }

    const uint32_t N = featureTable->POINTS_LENGTH;
    if (N == 0)
    {
        vsg::warn("Tiles3D::read_pnts(", filename, ") POINTS_LENGTH is 0.");
        return {};
    }

    // ---- positions -------------------------------------------------------
    std::vector<float> positions;
    positions.reserve(N * 3);

    if (featureTable->POSITION && featureTable->POSITION.values.size() >= N * 3)
    {
        positions.assign(featureTable->POSITION.values.begin(),
                         featureTable->POSITION.values.begin() + N * 3);
    }
    else if (featureTable->POSITION_QUANTIZED && featureTable->POSITION_QUANTIZED.values.size() >= N * 3)
    {
        // Quantized positions are meaningless without the volume: the spec
        // requires both globals when POSITION_QUANTIZED is used, and decoding
        // with a default volume would silently collapse the cloud to a 1m cube.
        if (featureTable->QUANTIZED_VOLUME_OFFSET.values.size() < 3 ||
            featureTable->QUANTIZED_VOLUME_SCALE.values.size() < 3)
        {
            vsg::warn("Tiles3D::read_pnts(", filename, ") POSITION_QUANTIZED without "
                      "QUANTIZED_VOLUME_OFFSET/SCALE; cannot decode.");
            return {};
        }
        const auto& off = featureTable->QUANTIZED_VOLUME_OFFSET.values;
        const auto& scl = featureTable->QUANTIZED_VOLUME_SCALE.values;
        const auto& q = featureTable->POSITION_QUANTIZED.values;
        for (uint32_t i = 0; i < N; ++i)
        {
            for (int c = 0; c < 3; ++c)
                positions.push_back(off[c] + (static_cast<float>(q[i * 3 + c]) / 65535.0f) * scl[c]);
        }
    }
    else
    {
        vsg::warn("Tiles3D::read_pnts(", filename, ") no POSITION or POSITION_QUANTIZED.");
        return {};
    }

    vsg::vec3 bbMin(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    vsg::vec3 bbMax(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
    for (uint32_t i = 0; i < N; ++i)
    {
        const vsg::vec3 p(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
        bbMin.x = std::min(bbMin.x, p.x); bbMin.y = std::min(bbMin.y, p.y); bbMin.z = std::min(bbMin.z, p.z);
        bbMax.x = std::max(bbMax.x, p.x); bbMax.y = std::max(bbMax.y, p.y); bbMax.z = std::max(bbMax.z, p.z);
    }

    // ---- colours ---------------------------------------------------------
    // All four spellings, because a file may use any one and a reader that
    // handles only RGBA renders most real point clouds black.
    std::vector<uint8_t> colours(static_cast<size_t>(N) * 4, 255);

    if (featureTable->RGBA && featureTable->RGBA.values.size() >= N * 4)
    {
        colours.assign(featureTable->RGBA.values.begin(),
                       featureTable->RGBA.values.begin() + N * 4);
    }
    else if (featureTable->RGB && featureTable->RGB.values.size() >= N * 3)
    {
        const auto& c = featureTable->RGB.values;
        for (uint32_t i = 0; i < N; ++i)
        {
            colours[i * 4 + 0] = c[i * 3 + 0];
            colours[i * 4 + 1] = c[i * 3 + 1];
            colours[i * 4 + 2] = c[i * 3 + 2];
            colours[i * 4 + 3] = 255;
        }
    }
    else if (featureTable->RGB565 && featureTable->RGB565.values.size() >= N)
    {
        // 5 bits red, 6 green, 5 blue. Expanded by replicating the high bits
        // into the low ones so that full-scale input maps to 255, not 248.
        const auto& c = featureTable->RGB565.values;
        for (uint32_t i = 0; i < N; ++i)
        {
            const uint16_t v = c[i];
            const uint8_t r5 = static_cast<uint8_t>((v >> 11) & 0x1F);
            const uint8_t g6 = static_cast<uint8_t>((v >> 5) & 0x3F);
            const uint8_t b5 = static_cast<uint8_t>(v & 0x1F);
            colours[i * 4 + 0] = static_cast<uint8_t>((r5 << 3) | (r5 >> 2));
            colours[i * 4 + 1] = static_cast<uint8_t>((g6 << 2) | (g6 >> 4));
            colours[i * 4 + 2] = static_cast<uint8_t>((b5 << 3) | (b5 >> 2));
            colours[i * 4 + 3] = 255;
        }
    }
    else if (featureTable->CONSTANT_RGBA && featureTable->CONSTANT_RGBA.values.size() >= 4)
    {
        const auto& c = featureTable->CONSTANT_RGBA.values;
        for (uint32_t i = 0; i < N; ++i)
        {
            colours[i * 4 + 0] = c[0];
            colours[i * 4 + 1] = c[1];
            colours[i * 4 + 2] = c[2];
            colours[i * 4 + 3] = c[3];
        }
    }
    // else: opaque white, already filled.

    const std::string glb = buildPointsGlb(positions, colours, bbMin, bbMax);

    if (const char* dbg = std::getenv("PNTS_DEBUG"); dbg && *dbg)
    {
        std::fprintf(stderr, "[pnts] N=%u glb=%zu bytes json=%.*s\n",
                     N, glb.size(), 900, glb.data() + 20);
    }

    auto opt = vsg::clone(options);
    opt->extensionHint = ".glb";

    vsg::mem_stream glb_fin(reinterpret_cast<const uint8_t*>(glb.data()), glb.size());
    auto model = vsg::read_cast<vsg::Node>(glb_fin, opt);
    if (!model)
    {
        vsg::warn("Tiles3D::read_pnts(", filename, ") built a glTF the reader would not accept.");
        return {};
    }

    if (featureTable->RTC_CENTER && featureTable->RTC_CENTER.values.size() == 3)
    {
        const auto& v = featureTable->RTC_CENTER.values;
        auto transform = vsg::MatrixTransform::create();
        transform->matrix = vsg::translate(vsg::dvec3(v[0], v[1], v[2]));
        transform->addChild(model);
        model = transform;
    }

    if (model && filename) model->setValue("pnts", filename);

    return model;
}
