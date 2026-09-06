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

#include <vsgXchange/Version.h>

#ifdef vsgXchange_draco
#    include <draco/compression/decode.h>
#    include <draco/point_cloud/point_cloud.h>
#endif

#include <cmath>
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
    else if (property == "BATCH_ID") parser.read_array(BATCH_ID);
    // Chain rather than dead-end on parser.warning(): the warning records a
    // message and consumes nothing, so an unrecognised value left in the
    // stream silently drops every property that follows it.
    else gltf::ExtensionsExtras::read_array(parser, property);
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
    // BATCH_ID is routed even though nothing downstream reads it. Leaving it
    // unrouted is not neutral: cesium-native's pointCloudBatched.pnts spells it
    // {"byteOffset":192,"componentType":"UNSIGNED_BYTE"} and POINTS_LENGTH
    // follows it, so the unconsumed object took POINTS_LENGTH with it and the
    // file read as "POINTS_LENGTH is 0" -- an empty cloud, no error.
    else if (property == "BATCH_ID") parser.read_object(BATCH_ID);
    else gltf::ExtensionsExtras::read_object(parser, property);
}

void Tiles3D::pnts_FeatureTable::read_number(vsg::JSONParser& parser, const std::string_view& property, std::istream& input)
{
    if (property == "POINTS_LENGTH") input >> POINTS_LENGTH;
    else if (property == "BATCH_LENGTH") input >> BATCH_LENGTH;
    else parser.warning();
}

void Tiles3D::pnts_FeatureTable::read_string(vsg::JSONParser& parser, const std::string_view&)
{
    // Consume and discard. An unrecognised string-valued property that is not
    // consumed truncates the rest of the feature table, and the symptom is a
    // file that reads as zero points rather than an error.
    std::string ignored;
    parser.read_string(ignored);
}

void Tiles3D::pnts_FeatureTable::convert()
{
    if (POINTS_LENGTH == 0 || !binary) return;

    // size_t, not uint32_t. POINTS_LENGTH comes from the file, and `3 * N` in
    // 32-bit arithmetic WRAPS: N = 1431655766 makes 3*N == 2. assign() would
    // then load two floats, the `values.size() >= N * 3` guard below would
    // compare against the same wrapped 2 and pass, and the bounding-box loop
    // would still run N times indexing positions[i * 3] far past the end.
    const size_t N = static_cast<size_t>(POINTS_LENGTH);
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
// 3DTILES_draco_point_compression
//
void Tiles3D::draco_point_compression::report(vsg::LogOutput& output)
{
    output.enter("3DTILES_draco_point_compression {");
    output("byteOffset = ", byteOffset);
    output("byteLength = ", byteLength);
    output.enter("properties = {");
    for (auto& [semantic, id] : properties.values) output("    ", semantic, ", ", id);
    output.leave();
    output.leave();
}

void Tiles3D::draco_point_compression::read_object(vsg::JSONParser& parser, const std::string_view& property)
{
    if (property == "properties")
        parser.read_object(properties);
    else
        gltf::ExtensionsExtras::read_object(parser, property);
}

void Tiles3D::draco_point_compression::read_number(vsg::JSONParser& parser, const std::string_view& property, std::istream& input)
{
    if (property == "byteOffset") input >> byteOffset;
    else if (property == "byteLength") input >> byteLength;
    else parser.warning();
}

namespace
{
#ifdef vsgXchange_draco
    //! Read one Draco attribute out as `components` floats per point.
    //!
    //! ConvertValue does the work of getting from whatever the attribute is
    //! stored as -- quantised integers, usually -- to float, which is what the
    //! generated glTF wants for POSITION and NORMAL alike.
    bool dracoToFloats(const draco::PointCloud& pc, uint32_t uniqueId,
                       size_t count, unsigned components, std::vector<float>& out)
    {
        const draco::PointAttribute* attr = pc.GetAttributeByUniqueId(uniqueId);
        if (!attr) return false;

        // The blob decides how many components it stored. Asking for more than
        // it has reads past the end of its value; fewer is a different
        // semantic. Either way it is not the attribute we were promised.
        if (attr->num_components() != components) return false;

        out.assign(count * components, 0.0f);
        for (size_t i = 0; i < count; ++i)
        {
            const auto index = attr->mapped_index(draco::PointIndex(static_cast<uint32_t>(i)));
            if (!attr->ConvertValue(index, static_cast<int8_t>(components),
                                    out.data() + i * components))
            {
                return false;
            }
        }
        return true;
    }

    //! The same, as bytes -- for RGB and RGBA, which are normalized u8.
    bool dracoToBytes(const draco::PointCloud& pc, uint32_t uniqueId,
                      size_t count, unsigned components, std::vector<uint8_t>& out)
    {
        const draco::PointAttribute* attr = pc.GetAttributeByUniqueId(uniqueId);
        if (!attr) return false;
        if (attr->num_components() != components) return false;

        out.assign(count * components, 0u);
        for (size_t i = 0; i < count; ++i)
        {
            const auto index = attr->mapped_index(draco::PointIndex(static_cast<uint32_t>(i)));
            if (!attr->ConvertValue(index, static_cast<int8_t>(components),
                                    out.data() + i * components))
            {
                return false;
            }
        }
        return true;
    }
#endif
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
                               const std::vector<float>& normals,        // 3 per point, or empty
                               const vsg::vec3& bbMin, const vsg::vec3& bbMax)
    {
        // size_t throughout: count * 16 in 32-bit arithmetic wraps well inside
        // the range of a declared POINTS_LENGTH, and the code would then append
        // the wrapped byte count while writing truncated GLB length fields.
        // read_pnts caps N before calling this, so the casts below are safe.
        const size_t count = positions.size() / 3;
        const size_t posBytes = count * 12;


        // COLOR_0 as FLOAT VEC4, not normalized UNSIGNED_BYTE. The compact
        // form is legal glTF and the reader accepts it, but the resulting
        // ubvec4 vertex buffer rendered every point pure black -- measured,
        // 20,469 of 20,469 non-background pixels exactly (0,0,0), against a
        // colour array that held the file's real values. Floats cost 16 bytes
        // a point instead of 4 and leave nothing to interpret.
        std::vector<float> fcolours;
        fcolours.reserve(colours.size());
        for (uint8_t c : colours) fcolours.push_back(static_cast<float>(c) / 255.0f);
        const size_t colBytesF = count * 16;

        const bool hasNormals = normals.size() >= count * 3 && count > 0;
        const size_t nrmBytes = hasNormals ? count * 12 : 0;

        std::string bin;
        bin.reserve(posBytes + colBytesF + nrmBytes + 4);
        bin.append(reinterpret_cast<const char*>(positions.data()), posBytes);
        bin.append(reinterpret_cast<const char*>(fcolours.data()), colBytesF);
        if (hasNormals) bin.append(reinterpret_cast<const char*>(normals.data()), nrmBytes);
        while (bin.size() % 4 != 0) bin.push_back('\0');   // chunks are 4-byte aligned

        // POSITION accessors REQUIRE min/max; a reader is entitled to reject the
        // asset without them, and vsgXchange uses them for the bound.
        char json[1600];

        if (hasNormals)
        {
            // A file that carries NORMAL or NORMAL_OCT16P is describing a
            // surface sampled as points -- a scan, a photogrammetry mesh
            // reduced to vertices -- and it wants shading. So this branch drops
            // KHR_materials_unlit and emits a third accessor, which is the only
            // reason the file bothered to store normals at all.
            std::snprintf(json, sizeof(json),
                "{\"asset\":{\"version\":\"2.0\"},\"scene\":0,"
                "\"scenes\":[{\"nodes\":[0]}],"
                "\"nodes\":[{\"mesh\":0,\"name\":\"points\"}],"
                "\"meshes\":[{\"primitives\":[{\"attributes\":"
                  "{\"POSITION\":0,\"COLOR_0\":1,\"NORMAL\":2},"
                "\"mode\":0,\"material\":0}]}],"
                "\"materials\":[{\"pbrMetallicRoughness\":{\"baseColorFactor\":[1,1,1,1],"
                "\"metallicFactor\":0,\"roughnessFactor\":1},\"doubleSided\":true}],"
                "\"accessors\":["
                  "{\"bufferView\":0,\"componentType\":5126,\"count\":%u,\"type\":\"VEC3\","
                   "\"min\":[%.9g,%.9g,%.9g],\"max\":[%.9g,%.9g,%.9g]},"
                  "{\"bufferView\":1,\"componentType\":5126,\"count\":%u,\"type\":\"VEC4\"},"
                  "{\"bufferView\":2,\"componentType\":5126,\"count\":%u,\"type\":\"VEC3\"}],"
                "\"bufferViews\":["
                  "{\"buffer\":0,\"byteOffset\":0,\"byteLength\":%u,\"target\":34962},"
                  "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34962},"
                  "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34962}],"
                "\"buffers\":[{\"byteLength\":%u}]}",
                (uint32_t)count, bbMin.x, bbMin.y, bbMin.z, bbMax.x, bbMax.y, bbMax.z,
                (uint32_t)count, (uint32_t)count,
                (uint32_t)posBytes,
                (uint32_t)posBytes, (uint32_t)colBytesF,
                (uint32_t)(posBytes + colBytesF), (uint32_t)nrmBytes,
                static_cast<uint32_t>(bin.size()));

            std::string js_n(json);
            while (js_n.size() % 4 != 0) js_n.push_back(' ');

            std::string glb_n;
            glb_n.append("glTF", 4);
            append_u32(glb_n, 2);
            append_u32(glb_n, static_cast<uint32_t>(12 + 8 + js_n.size() + 8 + bin.size()));
            append_u32(glb_n, static_cast<uint32_t>(js_n.size()));
            append_u32(glb_n, 0x4E4F534A);   // "JSON"
            glb_n += js_n;
            append_u32(glb_n, static_cast<uint32_t>(bin.size()));
            append_u32(glb_n, 0x004E4942);   // "BIN"
            glb_n += bin;
            return glb_n;
        }

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
            (uint32_t)count, bbMin.x, bbMin.y, bbMin.z, bbMax.x, bbMax.y, bbMax.z,
            (uint32_t)count, (uint32_t)posBytes, (uint32_t)posBytes, (uint32_t)colBytesF,
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

    // Validate the declared section lengths against the stream BEFORE using any
    // of them to size an allocation. Every one of these is attacker-controlled:
    // a 28-byte file with a valid "pnts" magic and
    // featureTableJSONByteLength = 0xFFFFFFFF would otherwise ask resize() for
    // 4 GiB of JSON that does not exist. The sums are done in uint64 because
    // adding four 32-bit lengths wraps.
    {
        const uint64_t sections =
            static_cast<uint64_t>(header.featureTableJSONByteLength) +
            header.featureTableBinaryByteLength +
            header.batchTableJSONByteLength +
            header.batchTableBinaryLength;

        const std::streampos here = fin.tellg();
        fin.seekg(0, std::ios::end);
        const std::streampos endPos = fin.tellg();
        fin.seekg(here);
        if (endPos < here) return {};
        const uint64_t remaining = static_cast<uint64_t>(endPos - here);

        if (sections > remaining)
        {
            vsg::warn("Tiles3D::read_pnts(", filename, ") declares ", sections,
                      " bytes of tables but only ", remaining, " remain.");
            return {};
        }
        if (header.byteLength != 0 &&
            static_cast<uint64_t>(header.byteLength) < sizeof(Header) + sections)
        {
            vsg::warn("Tiles3D::read_pnts(", filename, ") byteLength ",
                      header.byteLength, " is smaller than its own tables.");
            return {};
        }

        // And byteLength itself against the stream.
        //
        // The check above proves the TABLES fit; it says nothing about the
        // payload after them, which is sized as
        // `byteLength - sizeof(Header) - sections`. All four tables zero and a
        // byteLength near UINT32_MAX passes everything above and then asks
        // resize() for four gigabytes, on a worker thread, from a file that is
        // a few bytes long.
        if (header.byteLength != 0 &&
            static_cast<uint64_t>(header.byteLength) > sizeof(Header) + remaining)
        {
            vsg::warn("Tiles3D::read_pnts(", filename, ") byteLength ",
                      header.byteLength, " but only ", remaining,
                      " bytes follow the header.");
            return {};
        }
    }

    vsg::JSONParser parser;
    parser.options = options;

    // Without this the extension is stored as anonymous metadata and its
    // properties map -- the semantic-to-attribute-id table the decode needs --
    // is not reachable.
    parser.setObject("3DTILES_draco_point_compression", draco_point_compression::create());

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

    const size_t N = static_cast<size_t>(featureTable->POINTS_LENGTH);
    if (N == 0)
    {
        vsg::warn("Tiles3D::read_pnts(", filename, ") POINTS_LENGTH is 0.");
        return {};
    }

    // A declared point count is not a promise that the data exists. The
    // intermediate glTF holds 12 bytes of position and 16 of colour per point,
    // and its chunk lengths are 32-bit fields, so beyond this the generated
    // asset cannot even be described -- let alone allocated. 64M points is
    // ~1.9 GB of intermediate buffer, already far past anything real.
    constexpr size_t MAX_POINTS = 64u * 1024u * 1024u;
    if (N > MAX_POINTS)
    {
        vsg::warn("Tiles3D::read_pnts(", filename, ") POINTS_LENGTH ", N,
                  " exceeds the ", MAX_POINTS, " point limit.");
        return {};
    }

    // ---- Draco -----------------------------------------------------------
    //
    // 3DTILES_draco_point_compression puts every attribute in one compressed
    // blob and STILL lists POSITION, RGB and NORMAL in the feature table, each
    // with byteOffset 0 -- they all point at the head of the blob. So this has
    // to be decided BEFORE the semantics are read: reading them as written
    // produced a cloud whose extent came out [-1.5e+13, 3.4e+28] for a scene
    // that fits in a 5-metre box.
    std::vector<float> dracoPositions;
    std::vector<uint8_t> dracoColours;      // RGB or RGBA, 3 or 4 per point
    unsigned dracoColourComponents = 0;
    std::vector<float> dracoNormals;
    bool dracoDecoded = false;

    if (auto compression = featureTable->extension<draco_point_compression>(
            "3DTILES_draco_point_compression"))
    {
#ifdef vsgXchange_draco
        // The blob's extent comes from the file. Check it against the binary
        // section actually read before handing a pointer and a length to a
        // decoder that will trust both.
        const size_t binarySize = featureTable->binary ? featureTable->binary->size() : 0;
        const size_t offset = compression->byteOffset;
        const size_t length = compression->byteLength;

        if (length == 0 || offset > binarySize || length > binarySize - offset)
        {
            vsg::warn("Tiles3D::read_pnts(", filename, ") Draco blob at ", offset,
                      "+", length, " does not fit the ", binarySize,
                      " byte feature table binary.");
            return {};
        }

        // A compressed file reaches a large point count far more cheaply than
        // an uncompressed one: 64M points is 768 MB of POSITION to download
        // uncompressed, and a few hundred bytes of Draco. So the general
        // MAX_POINTS ceiling above is not the operative limit here. 8M points
        // is ~224 MB across position, colour and normal, and is already far
        // beyond a real point-cloud tile -- Cesium's own guidance is tens to
        // hundreds of thousands per tile.
        constexpr size_t MAX_DRACO_POINTS = 8u * 1024u * 1024u;
        if (N > MAX_DRACO_POINTS)
        {
            vsg::warn("Tiles3D::read_pnts(", filename, ") POINTS_LENGTH ", N,
                      " exceeds the ", MAX_DRACO_POINTS,
                      " point limit for a Draco-compressed cloud.");
            return {};
        }

        draco::DecoderBuffer decodeBuffer;
        decodeBuffer.Init(
            reinterpret_cast<const char*>(featureTable->binary->dataPointer()) + offset,
            length);

        draco::Decoder decoder;
        auto decoded = decoder.DecodePointCloudFromBuffer(&decodeBuffer);
        if (!decoded.ok())
        {
            vsg::warn("Tiles3D::read_pnts(", filename, ") Draco decode failed: ",
                      decoded.status().error_msg_string());
            return {};
        }

        const auto& cloud = *decoded.value();

        // The point count in the blob is the authority: POINTS_LENGTH is a
        // second, independent statement of the same number, and if the two
        // disagree the file is describing something that is not there.
        if (static_cast<size_t>(cloud.num_points()) != N)
        {
            vsg::warn("Tiles3D::read_pnts(", filename, ") POINTS_LENGTH is ", N,
                      " but the Draco blob holds ", cloud.num_points(), " points.");
            return {};
        }

        const auto& props = compression->properties.values;

        auto idOf = [&](const char* semantic, uint32_t& out) -> bool
        {
            auto i = props.find(semantic);
            if (i == props.end() || !i->second.valid()) return false;
            out = i->second.value;
            return true;
        };

        uint32_t id = 0;
        if (!idOf("POSITION", id) || !dracoToFloats(cloud, id, N, 3, dracoPositions))
        {
            vsg::warn("Tiles3D::read_pnts(", filename,
                      ") Draco blob has no usable POSITION attribute.");
            return {};
        }

        if (idOf("RGBA", id) && dracoToBytes(cloud, id, N, 4, dracoColours))
            dracoColourComponents = 4;
        else if (idOf("RGB", id) && dracoToBytes(cloud, id, N, 3, dracoColours))
            dracoColourComponents = 3;

        if (idOf("NORMAL", id))
        {
            // A normal that will not read is not a reason to lose the cloud;
            // an unlit cloud is a lesser failure than no cloud.
            if (!dracoToFloats(cloud, id, N, 3, dracoNormals))
                dracoNormals.clear();
        }

        // BATCH_ID is in `props` too. Nothing reads it yet, and it is listed
        // here rather than silently ignored so the next person can see that the
        // id is available and only the destination is missing.

        // Now DISCARD the uncompressed reading of everything the blob owns.
        //
        // convert() has already resolved POSITION, RGB and NORMAL against the
        // binary section, and for a compressed file their byteOffsets point
        // INTO the blob -- so those arrays hold compressed bytes reinterpreted
        // as floats and colour channels. Nothing below may fall back to them.
        // The one that bites is a semantic the extension claims but whose
        // attribute will not decode: the colour would silently become
        // compressed noise while the cloud still drew, which is the failure
        // this whole path exists to prevent.
        //
        // Only what `properties` NAMES is cleared. A partially compressed file
        // -- pointCloudDracoPartial.pnts compresses position alone and leaves
        // RGB and NORMAL uncompressed further down the same section -- must
        // keep the semantics the blob does not own.
        // Anything whose byteOffset lands INSIDE the blob is compressed data
        // however it is labelled. An attacker who simply omits RGB from
        // `properties` while leaving it declared at byteOffset 0 would
        // otherwise get the blob's header read as colour channels -- the same
        // failure, reached by not claiming the semantic rather than by
        // claiming it badly.
        const auto overlapsBlob = [&](uint32_t semanticOffset, bool present)
        {
            return present && semanticOffset >= offset && semanticOffset < offset + length;
        };
        if (overlapsBlob(featureTable->POSITION.byteOffset, (bool)featureTable->POSITION))
            featureTable->POSITION.values.clear();
        if (overlapsBlob(featureTable->POSITION_QUANTIZED.byteOffset, (bool)featureTable->POSITION_QUANTIZED))
            featureTable->POSITION_QUANTIZED.values.clear();
        if (overlapsBlob(featureTable->RGBA.byteOffset, (bool)featureTable->RGBA))
            featureTable->RGBA.values.clear();
        if (overlapsBlob(featureTable->RGB.byteOffset, (bool)featureTable->RGB))
            featureTable->RGB.values.clear();
        if (overlapsBlob(featureTable->RGB565.byteOffset, (bool)featureTable->RGB565))
            featureTable->RGB565.values.clear();
        if (overlapsBlob(featureTable->NORMAL.byteOffset, (bool)featureTable->NORMAL))
            featureTable->NORMAL.values.clear();
        if (overlapsBlob(featureTable->NORMAL_OCT16P.byteOffset, (bool)featureTable->NORMAL_OCT16P))
            featureTable->NORMAL_OCT16P.values.clear();

        for (const auto& [semantic, id] : props)
        {
            (void)id;
            if (semantic == "POSITION") featureTable->POSITION.values.clear();
            else if (semantic == "POSITION_QUANTIZED") featureTable->POSITION_QUANTIZED.values.clear();
            else if (semantic == "RGBA") featureTable->RGBA.values.clear();
            else if (semantic == "RGB") featureTable->RGB.values.clear();
            else if (semantic == "RGB565") featureTable->RGB565.values.clear();
            else if (semantic == "NORMAL") featureTable->NORMAL.values.clear();
            else if (semantic == "NORMAL_OCT16P") featureTable->NORMAL_OCT16P.values.clear();
            else if (semantic == "BATCH_ID") featureTable->BATCH_ID.values.clear();
        }

        dracoDecoded = true;
#else
        (void)compression;
        vsg::warn("Tiles3D::read_pnts(", filename, ") uses "
                  "3DTILES_draco_point_compression, and this build has no Draco.");
        return {};
#endif
    }

    // ---- positions -------------------------------------------------------
    std::vector<float> positions;
    positions.reserve(N * 3);

    if (dracoDecoded)
    {
        positions = std::move(dracoPositions);
    }
    else if (featureTable->POSITION && featureTable->POSITION.values.size() >= N * 3)
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

    if (dracoColourComponents == 4)
    {
        colours = std::move(dracoColours);
    }
    else if (dracoColourComponents == 3)
    {
        for (size_t i = 0; i < N; ++i)
        {
            colours[i * 4 + 0] = dracoColours[i * 3 + 0];
            colours[i * 4 + 1] = dracoColours[i * 3 + 1];
            colours[i * 4 + 2] = dracoColours[i * 3 + 2];
        }
    }
    else if (featureTable->RGBA && featureTable->RGBA.values.size() >= N * 4)
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

    // ---- normals ---------------------------------------------------------
    // Optional, and in two spellings. NORMAL_OCT16P packs a unit vector into
    // two bytes by octahedral encoding: it is not a truncated float triple, and
    // reading it as one yields normals that all point roughly the same way,
    // which shades a scanned surface as a flat sheet.
    std::vector<float> normals;
    if (!dracoNormals.empty())
    {
        normals = std::move(dracoNormals);
    }
    else if (featureTable->NORMAL && featureTable->NORMAL.values.size() >= N * 3)
    {
        normals.assign(featureTable->NORMAL.values.begin(),
                       featureTable->NORMAL.values.begin() + N * 3);
    }
    else if (featureTable->NORMAL_OCT16P && featureTable->NORMAL_OCT16P.values.size() >= N * 2)
    {
        const auto& o = featureTable->NORMAL_OCT16P.values;
        normals.reserve(N * 3);
        for (size_t i = 0; i < N; ++i)
        {
            // The 3D Tiles decode, matching cesium-native's
            // AttributeCompression::octDecodeInRange: both bytes to [-1,1],
            // then unfold the lower half of the octahedron, then renormalise.
            // Dropping the unfold looks almost right -- it is exact on the
            // upper hemisphere -- and mirrors every downward-facing normal.
            const float ex = static_cast<float>(o[i * 2 + 0]) / 255.0f * 2.0f - 1.0f;
            const float ey = static_cast<float>(o[i * 2 + 1]) / 255.0f * 2.0f - 1.0f;
            float nx = ex;
            float ny = ey;
            const float nz = 1.0f - (std::fabs(ex) + std::fabs(ey));
            if (nz < 0.0f)
            {
                const float oldx = nx;
                nx = (1.0f - std::fabs(ny)) * (oldx >= 0.0f ? 1.0f : -1.0f);
                ny = (1.0f - std::fabs(oldx)) * (ny >= 0.0f ? 1.0f : -1.0f);
            }
            const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
            const float inv = len > 0.0f ? 1.0f / len : 0.0f;
            normals.push_back(nx * inv);
            normals.push_back(ny * inv);
            normals.push_back(nz * inv);
        }
    }

    const std::string glb = buildPointsGlb(positions, colours, normals, bbMin, bbMax);

    if (const char* dbg = std::getenv("PNTS_DEBUG"); dbg && *dbg)
    {
        std::fprintf(stderr, "[pnts] N=%u glb=%zu bytes json=%.*s\n",
                     (unsigned)N, glb.size(), 900, glb.data() + 20);

        // The first two points, decoded. Extents alone cannot tell a correct
        // colour decode from a plausible one, and these are the numbers a
        // compressed file and its uncompressed twin have to agree on.
        for (size_t i = 0; i < N && i < 2; ++i)
        {
            std::fprintf(stderr, "[pnts]   p%zu pos=(%.5f,%.5f,%.5f) rgba=(%u,%u,%u,%u)",
                         i, positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2],
                         (unsigned)colours[i * 4], (unsigned)colours[i * 4 + 1],
                         (unsigned)colours[i * 4 + 2], (unsigned)colours[i * 4 + 3]);
            if (normals.size() >= (i + 1) * 3)
            {
                std::fprintf(stderr, " n=(%.5f,%.5f,%.5f)",
                             normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]);
            }
            std::fprintf(stderr, "\n");
        }
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
