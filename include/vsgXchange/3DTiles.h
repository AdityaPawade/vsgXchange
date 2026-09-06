#pragma once

/* <editor-fold desc="MIT License">

Copyright(c) 2025 Robert Osfield

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shimages be included in images
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

</editor-fold> */

#include <vsg/threading/OperationThreads.h>

#include <cstring>
#include <limits>

#include <vsgXchange/gltf.h>

namespace vsgXchange
{

    /// 3DTiles ReaderWriter, C++ won't handle class called 3DTiles so make do with Tiles3D.
    /// Specs for 3DTiles https://github.com/CesiumGS/3d-tiles
    class VSGXCHANGE_DECLSPEC Tiles3D : public vsg::Inherit<vsg::ReaderWriter, Tiles3D>
    {
    public:
        Tiles3D();

        vsg::ref_ptr<vsg::Object> read(const vsg::Path&, vsg::ref_ptr<const vsg::Options>) const override;
        vsg::ref_ptr<vsg::Object> read(std::istream&, vsg::ref_ptr<const vsg::Options>) const override;
        vsg::ref_ptr<vsg::Object> read(const uint8_t* ptr, size_t size, vsg::ref_ptr<const vsg::Options> options = {}) const override;

        vsg::ref_ptr<vsg::Object> read_json(std::istream&, vsg::ref_ptr<const vsg::Options>, const vsg::Path& filename = {}) const;
        vsg::ref_ptr<vsg::Object> read_b3dm(std::istream&, vsg::ref_ptr<const vsg::Options>, const vsg::Path& filename = {}) const;
        vsg::ref_ptr<vsg::Object> read_cmpt(std::istream&, vsg::ref_ptr<const vsg::Options>, const vsg::Path& filename = {}) const;
        vsg::ref_ptr<vsg::Object> read_i3dm(std::istream&, vsg::ref_ptr<const vsg::Options>, const vsg::Path& filename = {}) const;
        vsg::ref_ptr<vsg::Object> read_pnts(std::istream&, vsg::ref_ptr<const vsg::Options>, const vsg::Path& filename = {}) const;
        vsg::ref_ptr<vsg::Object> read_tiles(const vsg::Path& filename, vsg::ref_ptr<const vsg::Options> options) const;

        vsg::Logger::Level level = vsg::Logger::LOGGER_WARN;

        bool supportedExtension(const vsg::Path& ext) const;

        bool getFeatures(Features& features) const override;

        static constexpr const char* report = "report";                 /// bool, report parsed glTF to console, defaults to false
        static constexpr const char* instancing = "instancing";         /// bool, hint for using vsg::InstanceNode/InstanceDraw for instancing where possible.
        static constexpr const char* pixel_ratio = "pixel_ratio";       /// double, sets the SceneGraphBuilder::pixelErrorToScreenHeightRatio value used for setting LOD ranges.
        static constexpr const char* pre_load_level = "pre_load_level"; /// uint, sets the SceneGraphBuilder::preLoadLevel values to control what LOD level are pre loaded when reading a tileset.

        bool readOptions(vsg::Options& options, vsg::CommandLine& arguments) const override;

    public:
        /// https://github.com/CesiumGS/3d-tiles/blob/1.0/specification/schema/boundingVolume.schema.json
        struct VSGXCHANGE_DECLSPEC BoundingVolume : public vsg::Inherit<gltf::ExtensionsExtras, BoundingVolume>
        {
            vsg::ValuesSchema<double> box;
            vsg::ValuesSchema<double> region;
            vsg::ValuesSchema<double> sphere;

            void read_array(vsg::JSONParser& parser, const std::string_view& property) override;

            void report(vsg::LogOutput& output);
        };

        /// https://github.com/CesiumGS/3d-tiles/blob/1.0/specification/schema/tile.content.schema.json
        struct VSGXCHANGE_DECLSPEC Content : public vsg::Inherit<gltf::ExtensionsExtras, Content>
        {
            vsg::ref_ptr<BoundingVolume> boundingVolume;
            std::string uri;

            // loaded from uri
            vsg::ref_ptr<vsg::Object> object;

            void read_object(vsg::JSONParser& parser, const std::string_view& property) override;
            void read_string(vsg::JSONParser& parser, const std::string_view& property) override;

            void report(vsg::LogOutput& output);
        };

        /// https://github.com/CesiumGS/3d-tiles/blob/1.0/specification/schema/tile.schema.json
        struct VSGXCHANGE_DECLSPEC Tile : public vsg::Inherit<gltf::ExtensionsExtras, Tile>
        {
            vsg::ref_ptr<BoundingVolume> boundingVolume;
            vsg::ref_ptr<BoundingVolume> viewerRequestVolume;
            double geometricError = 0.0;
            std::string refine;
            vsg::ValuesSchema<double> transform;
            vsg::ObjectsSchema<Tile> children;
            vsg::ref_ptr<Content> content;

            void read_array(vsg::JSONParser& parser, const std::string_view& property) override;
            void read_object(vsg::JSONParser& parser, const std::string_view& property) override;
            void read_number(vsg::JSONParser& parser, const std::string_view& property, std::istream& input) override;
            void read_string(vsg::JSONParser& parser, const std::string_view& property) override;

            void report(vsg::LogOutput& output);
        };

        /// https://github.com/CesiumGS/3d-tiles/blob/1.0/specification/schema/properties.schema.json
        struct VSGXCHANGE_DECLSPEC PropertyRange : public vsg::Inherit<gltf::ExtensionsExtras, PropertyRange>
        {
            double minimum = 0.0;
            double maximum = 0.0;

            void read_number(vsg::JSONParser& parser, const std::string_view& property, std::istream& input) override;
        };

        struct VSGXCHANGE_DECLSPEC Properties : public vsg::Inherit<gltf::ExtensionsExtras, Properties>
        {
            std::map<std::string, PropertyRange> properties;

            void read_object(vsg::JSONParser& parser, const std::string_view& property_name) override;

            void report(vsg::LogOutput& output);
        };

        /// https://github.com/CesiumGS/3d-tiles/blob/main/specification/schema/asset.schema.json
        struct VSGXCHANGE_DECLSPEC Asset : public vsg::Inherit<gltf::ExtensionsExtras, Asset>
        {
            std::string version;
            std::string tilesetVersion;

            std::map<std::string, std::string> strings;
            std::map<std::string, double> numbers;

            void read_string(vsg::JSONParser& parser, const std::string_view& property) override;
            void read_number(vsg::JSONParser& parser, const std::string_view& property, std::istream& input) override;
            void report(vsg::LogOutput& output);
        };

        /// https://github.com/CesiumGS/3d-tiles/blob/main/specification/schema/tileset.schema.json
        struct VSGXCHANGE_DECLSPEC Tileset : public vsg::Inherit<gltf::ExtensionsExtras, Tileset>
        {
            vsg::ref_ptr<Asset> asset;
            vsg::ref_ptr<Properties> properties;
            vsg::ref_ptr<Tile> root;
            double geometricError = 0.0;
            vsg::StringsSchema extensionsUsed;
            vsg::StringsSchema extensionsRequired;

            void read_array(vsg::JSONParser& parser, const std::string_view& property) override;
            void read_object(vsg::JSONParser& parser, const std::string_view& property) override;
            void read_number(vsg::JSONParser& parser, const std::string_view& property, std::istream& input) override;

            void report(vsg::LogOutput& output);

            virtual void resolveURIs(vsg::ref_ptr<const vsg::Options> options);
        };

        /// Template class for reading an array of values from JSON or from a binary block
        template<typename T>
        struct ArraySchema : public Inherit<vsg::JSONParser::Schema, ArraySchema<T>>
        {
            const uint32_t invalidOffset = std::numeric_limits<uint32_t>::max();
            uint32_t byteOffset = invalidOffset;
            std::vector<T> values;

            void read_number(vsg::JSONParser&, std::istream& input) override
            {
                T value;
                input >> value;
                values.push_back(value);
            }

            void read_number(vsg::JSONParser& parser, const std::string_view& property, std::istream& input) override
            {
                if (property == "byteOffset")
                    input >> byteOffset;
                else
                    parser.warning();
            }

            /// Consume and discard "componentType" and "type".
            ///
            /// This override looks pointless -- the value is thrown away and T
            /// is already fixed by the schema -- and it is load-bearing. A
            /// feature-table semantic is written
            ///
            ///     "NORMAL_UP":{"byteOffset":12000,"componentType":"FLOAT","type":"VEC3"}
            ///
            /// and without a read_string here the base Schema never advances
            /// past the "FLOAT", so the parser loses its place and EVERY
            /// property after the first one is silently dropped. The failure is
            /// invisible: POSITION parses (its byteOffset precedes the first
            /// string), so instances are positioned correctly, while NORMAL_UP,
            /// NORMAL_RIGHT and SCALE_NON_UNIFORM keep byteOffset == invalid,
            /// assign() returns early, and every instance falls back to an
            /// identity rotation and unit scale. A city of instanced wall
            /// panels then renders as a flat carpet of horizontal slabs with no
            /// error, no warning, and correct-looking geometry everywhere else.
            void read_string(vsg::JSONParser& parser, const std::string_view&) override
            {
                std::string ignored;
                parser.read_string(ignored);
            }

            /// Copy `count` values from the feature-table binary at byteOffset.
            ///
            /// Everything here comes from the FILE -- byteOffset from its JSON,
            /// count from a length the file declares -- and a 3D Tiles payload
            /// is untrusted content fetched over HTTP. This used to compute
            /// `reinterpret_cast<T*>(binary.data() + byteOffset)` and read
            /// `count` elements with no validation at all, so
            ///
            ///     {"POINTS_LENGTH":1,"POSITION":{"byteOffset":0}}
            ///
            /// against a 1-byte binary section read a 4-byte float out of a
            /// 1-byte allocation, and a byteOffset near UINT32_MAX formed an
            /// invalid pointer before it was ever dereferenced.
            ///
            /// memcpy rather than a typed dereference: byteOffset need not be a
            /// multiple of sizeof(T), and an unaligned load is undefined
            /// behaviour, not merely slow.
            void assign(vsg::ubyteArray& binary, size_t count)
            {
                if (!values.empty() || byteOffset == invalidOffset) return;

                const size_t total = binary.size();
                const size_t offset = static_cast<size_t>(byteOffset);
                if (offset > total) return;

                // Checked: count * sizeof(T) must fit both in size_t and in
                // what remains after the offset.
                if (count > (std::numeric_limits<size_t>::max() / sizeof(T))) return;
                const size_t bytes = count * sizeof(T);
                if (bytes > total - offset) return;

                values.resize(count);
                if (count > 0) std::memcpy(values.data(), binary.data() + offset, bytes);
            }

            explicit operator bool() const noexcept { return !values.empty(); }
        };

        struct VSGXCHANGE_DECLSPEC b3dm_FeatureTable : public vsg::Inherit<gltf::ExtensionsExtras, b3dm_FeatureTable>
        {
            // storage for binary section
            vsg::ref_ptr<vsg::ubyteArray> binary;

            uint32_t BATCH_LENGTH = 0;
            ArraySchema<float> RTC_CENTER;
            void read_array(vsg::JSONParser& parser, const std::string_view& property) override;
            void read_object(vsg::JSONParser& parser, const std::string_view& property) override;
            void read_number(vsg::JSONParser& parser, const std::string_view& property, std::istream& input) override;

            void convert();

            void report(vsg::LogOutput& output);
        };

        /// https://github.com/CesiumGS/3d-tiles/blob/1.0/specification/TileFormats/PointCloud
        ///
        /// A .pnts feature table. Positions are either FLOAT32 VEC3 or, when
        /// POSITION_QUANTIZED is used, UINT16 VEC3 that must be scaled by
        /// QUANTIZED_VOLUME_SCALE/65535 and offset by QUANTIZED_VOLUME_OFFSET.
        ///
        /// Colour has four spellings and a file may use any one of them:
        /// RGBA (ubyte x4), RGB (ubyte x3), RGB565 (a single uint16 with 5/6/5
        /// bit channels) or the global CONSTANT_RGBA. A reader that handles
        /// only RGBA silently renders most real point clouds black.
        /// 3DTILES_draco_point_compression, on a .pnts feature table.
        ///
        /// One Draco-compressed blob in the feature table's binary section
        /// carries every attribute, and `properties` maps each semantic to the
        /// blob's own attribute id. The semantics are still listed beside it in
        /// the feature table, each with byteOffset 0 -- they all point at the
        /// head of the blob -- so a reader that ignores this extension reads
        /// compressed bytes as floats and produces a cloud whose extent came
        /// out [-1.5e+13, 3.4e+28] for a scene that fits in a 5-metre box.
        ///
        /// Not the same extension as glTF's KHR_draco_mesh_compression: that
        /// one names a bufferView and decodes a MESH; this names a byte range
        /// and decodes a POINT CLOUD, which has no faces.
        struct VSGXCHANGE_DECLSPEC draco_point_compression : public vsg::Inherit<gltf::ExtensionsExtras, draco_point_compression>
        {
            /// Semantic ("POSITION", "RGB", ...) to Draco attribute unique id.
            gltf::Attributes properties;

            /// Where the compressed blob sits in the feature table's binary.
            uint32_t byteOffset = 0;
            uint32_t byteLength = 0;

            // The prototype registered with the parser is cloned per use.
            vsg::ref_ptr<vsg::Object> clone(const vsg::CopyOp&) const override
            {
                return draco_point_compression::create(*this);
            }

            void report(vsg::LogOutput& output);
            void read_object(vsg::JSONParser& parser, const std::string_view& property) override;
            void read_number(vsg::JSONParser& parser, const std::string_view& property, std::istream& input) override;
        };

        struct VSGXCHANGE_DECLSPEC pnts_FeatureTable : public vsg::Inherit<gltf::ExtensionsExtras, pnts_FeatureTable>
        {
            // storage for binary section
            vsg::ref_ptr<vsg::ubyteArray> binary;

            uint32_t POINTS_LENGTH = 0;
            uint32_t BATCH_LENGTH = 0;

            ArraySchema<float> POSITION;
            ArraySchema<uint16_t> POSITION_QUANTIZED;
            ArraySchema<uint8_t> RGBA;
            ArraySchema<uint8_t> RGB;
            ArraySchema<uint16_t> RGB565;
            ArraySchema<uint8_t> CONSTANT_RGBA;
            ArraySchema<float> NORMAL;
            ArraySchema<uint8_t> NORMAL_OCT16P;
            ArraySchema<float> RTC_CENTER;
            ArraySchema<float> QUANTIZED_VOLUME_OFFSET;
            ArraySchema<float> QUANTIZED_VOLUME_SCALE;

            /// Declared so the JSON is CONSUMED, not because it is decoded yet.
            ///
            /// Its width is variable -- the spec allows UNSIGNED_BYTE, SHORT or
            /// INT via componentType -- and there is nowhere to put per-point
            /// batch ids until the reader carries feature IDs through. But
            /// leaving it out is not neutral: an unroutered property means its
            /// object body is parsed against the feature table itself, the
            /// string "UNSIGNED_BYTE" desyncs the parser, and everything after
            /// it in the document is lost. In cesium-native's
            /// pointCloudBatched.pnts, POINTS_LENGTH sits after BATCH_ID, so
            /// the whole file read as zero points.
            ArraySchema<uint8_t> BATCH_ID;

            void read_array(vsg::JSONParser& parser, const std::string_view& property) override;
            void read_object(vsg::JSONParser& parser, const std::string_view& property) override;
            void read_number(vsg::JSONParser& parser, const std::string_view& property, std::istream& input) override;

            /// Swallow the value of any string property this schema does not
            /// know. Without it an unrecognised semantic -- a future one, a
            /// vendor extension, anything -- silently truncates the rest of the
            /// feature table rather than simply being ignored.
            void read_string(vsg::JSONParser& parser, const std::string_view& property) override;

            void convert();

            void report(vsg::LogOutput& output);
        };

        struct BatchTable;

        struct VSGXCHANGE_DECLSPEC Batch : public vsg::Inherit<vsg::JSONtoMetaDataSchema, Batch>
        {
            uint32_t byteOffset = 0;
            std::string componentType;
            std::string type;

            void convert(BatchTable& batchTable);

            // read array parts
            void read_number(vsg::JSONParser& parser, std::istream& input) override;

            // read object parts
            void read_string(vsg::JSONParser& parser, const std::string_view& property) override;
            void read_number(vsg::JSONParser& parser, const std::string_view& property, std::istream& input) override;
        };

        /// Per-feature properties for one tile, in one shape whatever wrote them.
        ///
        /// 3D Tiles carries this two different ways and the application should
        /// not have to know which it got. A 1.0 tile has a BATCH TABLE -- JSON
        /// arrays, one entry per feature, beside a `_BATCHID` vertex attribute.
        /// A 1.1 tile has EXT_structural_metadata property tables selected by
        /// `_FEATURE_ID_n`. Both mean the same thing: this tile draws N objects,
        /// and each has these named values.
        ///
        /// Attached to the tile's node under FEATURE_TABLE_KEY, beside the
        /// per-vertex ids under FEATURE_IDS_KEY. Together those two answer
        /// "what did the operator just click on".
        class VSGXCHANGE_DECLSPEC FeatureTable : public vsg::Inherit<vsg::Object, FeatureTable>
        {
        public:
            /// How many features this tile draws. Every array in `properties`
            /// has this many entries.
            uint32_t count = 0;

            /// Property name to its values, one per feature, in feature order.
            /// The concrete type is whatever the document used -- a stringArray
            /// for names, a doubleArray for heights -- so a caller reads it with
            /// the usual vsg::Data visitors rather than a variant of our own.
            std::map<std::string, vsg::ref_ptr<vsg::Data>> properties;

            /// The value of `name` for feature `index`, formatted for display,
            /// or an empty string if there is no such property or the index is
            /// outside the table. Never throws: the index comes from a picked
            /// triangle and the name from an HTTP request.
            std::string valueAsString(const std::string& name, uint32_t index) const;
        };

        struct VSGXCHANGE_DECLSPEC BatchTable : public vsg::Inherit<gltf::ExtensionsExtras, BatchTable>
        {
            std::map<std::string, vsg::ref_ptr<Batch>> batches;

            uint32_t length = 0;
            vsg::ref_ptr<vsg::ubyteArray> binary;

            void read_array(vsg::JSONParser& parser, const std::string_view& property) override;
            void read_object(vsg::JSONParser& parser, const std::string_view& property) override;

            void convert();

            void report(vsg::LogOutput& output);
        };

        /// Where a tile's per-feature metadata is attached on its node.
        ///
        /// Named constants because three libraries read them -- the reader
        /// writes, the SDK's pick reads, and the tests assert -- and a typo in
        /// any one of them is a silent "this tile has no metadata".
        static constexpr const char* FEATURE_TABLE_KEY = "tiles3d.featureTable";
        static constexpr const char* FEATURE_IDS_KEY = "tiles3d.featureIds";

        /// The batch table of a b3dm payload, without reading its geometry.
        ///
        /// For a host that unpacks the b3dm wrapper itself and hands only the
        /// embedded glb to a reader -- which rocky does deliberately, to keep
        /// the header arithmetic in bounds -- the batch table would otherwise
        /// be discarded with the wrapper, and the tile would draw with no way
        /// to ask what is in it.
        ///
        /// Bounded: a header describing more than \p size holds is refused
        /// rather than believed, so nothing here allocates on a number an
        /// untrusted server chose.
        ///
        /// eturn The table, or null when the payload is not a b3dm, is
        ///         inconsistent, or simply carries no batch table -- all three
        ///         are ordinary and none is an error.
        static vsg::ref_ptr<FeatureTable> readFeatureTable(const uint8_t* b3dm, size_t size);

        /// Put \p table on \p model's root and on every node carrying
        /// FEATURE_IDS_KEY.
        ///
        /// Both places, because a consumer searching for the two separately
        /// pairs whatever it finds first -- which can be a nested child tile's
        /// ids with an ancestor's table, showing the wrong feature's data
        /// confidently and with no warning. Having them on one node makes that
        /// impossible rather than merely unlikely.
        ///
        /// eturn How many id-carrying nodes were paired. Zero means the
        ///         payload has a batch table but no _BATCHID, so its features
        ///         cannot be picked.
        static size_t attachFeatureTable(vsg::Node& model, vsg::ref_ptr<FeatureTable> table);

        // https://github.com/CesiumGS/3d-tiles/blob/main/specification/TileFormats/Instanced3DModel/README.adoc
        struct VSGXCHANGE_DECLSPEC i3dm_FeatureTable : public vsg::Inherit<gltf::ExtensionsExtras, i3dm_FeatureTable>
        {
            // storage for binary section
            vsg::ref_ptr<vsg::ubyteArray> binary;

            // Instance sematics
            ArraySchema<float> POSITION;
            ArraySchema<uint16_t> POSITION_QUANTIZED;
            ArraySchema<float> NORMAL_UP;
            ArraySchema<float> NORMAL_RIGHT;
            ArraySchema<uint16_t> NORMAL_UP_OCT32P;
            ArraySchema<uint16_t> NORMAL_RIGHT_OCT32P;
            ArraySchema<float> SCALE;
            ArraySchema<float> SCALE_NON_UNIFORM;
            ArraySchema<uint32_t> BATCH_ID;

            // Global sematics
            uint32_t INSTANCES_LENGTH = 0;
            ArraySchema<float> RTC_CENTER;
            ArraySchema<float> QUANTIZED_VOLUME_OFFSET;
            ArraySchema<float> QUANTIZED_VOLUME_SCALE;
            bool EAST_NORTH_UP = false;

            void read_array(vsg::JSONParser& parser, const std::string_view& property) override;
            void read_object(vsg::JSONParser& parser, const std::string_view& property) override;
            void read_number(vsg::JSONParser& parser, const std::string_view& property, std::istream& input) override;
            void read_bool(vsg::JSONParser& parser, const std::string_view& property, bool value) override;

            void convert();

            void report(vsg::LogOutput& output);
        };

    public:
        class VSGXCHANGE_DECLSPEC SceneGraphBuilder : public vsg::Inherit<vsg::Object, SceneGraphBuilder>
        {
        public:
            SceneGraphBuilder();

            vsg::ref_ptr<vsg::Options> options;
            vsg::ref_ptr<vsg::ShaderSet> shaderSet;
            vsg::ref_ptr<vsg::SharedObjects> sharedObjects;
            vsg::ref_ptr<vsg::OperationThreads> operationThreads;

            vsg::ref_ptr<vsg::EllipsoidModel> ellipsoidModel = vsg::EllipsoidModel::create();
            vsg::CoordinateConvention source_coordinateConvention = vsg::CoordinateConvention::Y_UP;
            double pixelErrorToScreenHeightRatio = 16.0/1024.0; // 16 pixel error on a 1024 height viewport.
            uint32_t preLoadLevel = 1;

            virtual void assignResourceHints(vsg::ref_ptr<vsg::Node> node);

            virtual double computeScreenHeightRatio(const vsg::dsphere& bound, double geometricError) const;

            virtual vsg::dmat4 createMatrix(const std::vector<double>& values);
            virtual vsg::dsphere createBound(vsg::ref_ptr<BoundingVolume> boundingVolume);
            virtual vsg::ref_ptr<vsg::Node> readTileChildren(vsg::ref_ptr<Tiles3D::Tile> tile, uint32_t level, const std::string& inherited_refine);
            virtual vsg::ref_ptr<vsg::Node> createTile(vsg::ref_ptr<Tiles3D::Tile> tile, uint32_t level, const std::string& inherited_refine);
            virtual vsg::ref_ptr<vsg::Object> createSceneGraph(vsg::ref_ptr<Tiles3D::Tileset> tileset, vsg::ref_ptr<const vsg::Options> in_options);
        };
    };

} // namespace vsgXchange

EVSG_type_name(vsgXchange::Tiles3D)
