/* <editor-fold desc="MIT License">

Copyright(c) 2025 Robert Osfield

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

</editor-fold> */

#include <vsgXchange/3DTiles.h>

#include <cstring>

#include <vsg/io/Path.h>
#include <vsg/io/mem_stream.h>
#include <vsg/io/read.h>
#include <vsg/io/write.h>
#include <vsg/nodes/MatrixTransform.h>
#include <vsg/threading/OperationThreads.h>
#include <vsg/utils/CommandLine.h>

#include <fstream>
#include <iostream>
#include <stack>

using namespace vsgXchange;

vsg::ref_ptr<vsg::Object> Tiles3D::read_cmpt(std::istream& fin, vsg::ref_ptr<const vsg::Options> options, const vsg::Path& filename) const
{
    fin.seekg(0);

    // https://github.com/CesiumGS/3d-tiles/blob/main/specification/TileFormats/Composite/README.adoc
    struct Header
    {
        char magic[4] = {0, 0, 0, 0};
        uint32_t version = 0;
        uint32_t byteLength = 0;
        uint32_t tilesLength = 0;
    };

    struct InnerHeader
    {
        char magic[4] = {0, 0, 0, 0};
        uint32_t version = 0;
        uint32_t byteLength = 0;
    };

    Header header;
    fin.read(reinterpret_cast<char*>(&header), sizeof(Header));

    if (!fin.good())
    {
        vsg::warn("IO error reading cmpt file.");
        return {};
    }

    if (strncmp(header.magic, "cmpt", 4) != 0)
    {
        vsg::warn("magic number not cmpt");
        return {};
    }

    // byteLength is the FILE's number. Subtracting the header from it in
    // unsigned arithmetic wraps for anything under 16, and resize() then asks
    // for about four gigabytes.
    if (header.byteLength < sizeof(Header))
    {
        vsg::warn("cmpt byteLength ", header.byteLength, " is smaller than its own header.");
        return {};
    }

    // A composite is a wrapper: its inner tiles are the payload, so a bound on
    // how many it may declare is a bound on the work one file can ask for.
    // Real composites hold a handful.
    constexpr uint32_t MAX_INNER_TILES = 4096;
    if (header.tilesLength > MAX_INNER_TILES)
    {
        vsg::warn("cmpt declares ", header.tilesLength, " inner tiles, more than the ",
                  MAX_INNER_TILES, " limit.");
        return {};
    }

    const uint32_t sizeOfTiles = header.byteLength - static_cast<uint32_t>(sizeof(Header));
    std::string binary;
    binary.resize(sizeOfTiles);
    fin.read(binary.data(), sizeOfTiles);
    if (!fin.good())
    {
        vsg::warn("IO error reading cmpt file.");
        return {};
    }

    auto group = vsg::Group::create();
    std::list<InnerHeader> innerHeaders;
    uint32_t pos = 0;
    for (uint32_t i = 0; i < header.tilesLength; ++i)
    {
        // Every one of these checks is about a number that came from the file.
        // Without them `&binary[pos]` walks off the end of the buffer -- pos
        // advances by an inner byteLength the file chose, and tilesLength says
        // how many times to do it. A composite naming 1000 inner tiles in a
        // 100-byte payload read a kilobyte past its own allocation.
        if (pos > binary.size() || binary.size() - pos < sizeof(InnerHeader))
        {
            vsg::warn("cmpt inner tile ", i, " starts past the end of the payload.");
            break;
        }

        InnerHeader tileHeader;
        std::memcpy(&tileHeader, binary.data() + pos, sizeof(InnerHeader));
        const InnerHeader* tile = &tileHeader;
        innerHeaders.push_back(tileHeader);

        // A zero-length inner tile would leave pos where it is: not an endless
        // loop, since tilesLength bounds the iterations, but it would decode
        // the same bytes over and over and is malformed either way.
        if (tile->byteLength < sizeof(InnerHeader) ||
            tile->byteLength > binary.size() - pos)
        {
            vsg::warn("cmpt inner tile ", i, " declares ", tile->byteLength,
                      " bytes, of which ", binary.size() - pos, " remain.");
            break;
        }

        vsg::mem_stream binary_fin(reinterpret_cast<uint8_t*>(&binary[pos]), tile->byteLength);
        pos += tile->byteLength;

        std::string ext = ".";
        for (int c = 0; c < 4; ++c)
        {
            if (tile->magic[c] != 0)
                ext.push_back(tile->magic[c]);
            else
                break;
        }

        auto opt = vsg::clone(options);
        opt->extensionHint = ext;

#if 0
        // force axis to Z up.
        auto upAxis = vsg::CoordinateConvention::Z_UP;
        opt->formatCoordinateConventions[".gltf"] = upAxis;
        opt->formatCoordinateConventions[".glb"] = upAxis;
#endif

        if (auto model = vsg::read_cast<vsg::Node>(binary_fin, opt))
        {
            group->addChild(model);
        }
    }

    if (vsg::value<bool>(false, gltf::report, options))
    {
        vsg::LogOutput output;

        output("Tiles3D::read_cmpt(..)");
        output("magic = ", header.magic);
        output("version = ", header.version);
        output("byteLength = ", header.byteLength);
        output("tilesLength = ", header.tilesLength);

        output("innerHeaders.size() = ", innerHeaders.size());
        for (auto& innerHeader : innerHeaders)
        {
            output("   {", innerHeader.magic, ", ", innerHeader.version, ", ", innerHeader.byteLength, " }");
        }
    }

    vsg::ref_ptr<vsg::Node> model;

    if (group->children.size() == 1)
        model = group->children[0];
    else if (!group->children.empty())
        model = group;

    if (model && filename) model->setValue("b3dm", filename);

    return model;
}
