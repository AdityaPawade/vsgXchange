/* <editor-fold desc="MIT License">

Copyright(c) 2025 Robert Osfield

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

</editor-fold> */

#include <vsg/core/Auxiliary.h>
#include <set>

#include <limits>

#include <vsgXchange/3DTiles.h>
#include <vsgXchange/gltf.h>

#include <vsg/animation/AnimationGroup.h>
#include <vsg/animation/Joint.h>
#include <vsg/animation/JointSampler.h>
#include <vsg/animation/TransformSampler.h>
#include <vsg/lighting/DirectionalLight.h>
#include <vsg/lighting/PointLight.h>
#include <vsg/lighting/SpotLight.h>
#include <vsg/maths/transform.h>
#include <vsg/nodes/CullGroup.h>
#include <vsg/nodes/CullNode.h>
#include <vsg/nodes/DepthSorted.h>
#include <vsg/nodes/Group.h>
#include <vsg/nodes/InstanceDraw.h>
#include <vsg/nodes/InstanceDrawIndexed.h>
#include <vsg/nodes/Layer.h>
#include <vsg/nodes/MatrixTransform.h>
#include <vsg/nodes/StateGroup.h>
#include <vsg/nodes/Switch.h>
#include <vsg/nodes/VertexDraw.h>
#include <vsg/nodes/VertexIndexDraw.h>
#include <vsg/state/material.h>
#include <vsg/utils/ComputeBounds.h>
#include <vsg/utils/GraphicsPipelineConfigurator.h>

#include <vsg/io/write.h>

#include <cmath>   // std::pow, for the base-colour factor encode

#ifdef vsgXchange_draco
#    include "draco/compression/decode.h"
#    include "draco/core/decoder_buffer.h"
#endif

using namespace vsgXchange;

gltf::SceneGraphBuilder::SceneGraphBuilder()
{
    attributeLookup = {
        {"POSITION", "vsg_Vertex"},
        {"NORMAL", "vsg_Normal"},
        {"TEXCOORD_0", "vsg_TexCoord0"},
        {"TEXCOORD_1", "vsg_TexCoord1"},
        {"TEXCOORD_2", "vsg_TexCoord2"},
        {"TEXCOORD_3", "vsg_TexCoord3"},
        {"COLOR", "vsg_Color"},
        {"COLOR_0", "vsg_Color"},
        {"JOINTS_0", "vsg_JointIndices"},
        {"WEIGHTS_0", "vsg_JointWeights"},
        {"TRANSLATION", "vsg_Translation"},
        {"ROTATION", "vsg_Rotation"},
        {"SCALE", "vsg_Scale"}};
}

void gltf::SceneGraphBuilder::assign_extras(ExtensionsExtras& src, vsg::Object& dest)
{
    if (src.extras)
    {
        if (src.extras->object)
        {
            dest.setObject("extras", src.extras->object);
        }
        else if (src.extras->objects)
        {
            dest.setObject("extras", src.extras->objects);
        }
    }
};

void gltf::SceneGraphBuilder::assign_name_extras(NameExtensionsExtras& src, vsg::Object& dest)
{
    if (!src.name.empty())
    {
        if (auto joint = dest.cast<vsg::Joint>())
            joint->name = src.name;
        else if (auto animation = dest.cast<vsg::Animation>())
            animation->name = src.name;
        else if (auto animationSampler = dest.cast<vsg::AnimationSampler>())
            animationSampler->name = src.name;
        else
            dest.setValue("name", src.name);
    }

    if (src.extras)
    {
        if (src.extras->object)
        {
            vsg::info("Assignig extras object ", src.extras->object);
            dest.setObject("extras", src.extras->object);
        }
        else if (src.extras->objects)
        {
            vsg::info("Assignig extras objects ", src.extras->object);
            dest.setObject("extras", src.extras->objects);
        }
    }
};

vsg::ref_ptr<vsg::Data> gltf::SceneGraphBuilder::createBuffer(vsg::ref_ptr<gltf::Buffer> gltf_buffer)
{
    return gltf_buffer->data;
}

vsg::ref_ptr<vsg::Data> gltf::SceneGraphBuilder::createBufferView(vsg::ref_ptr<gltf::BufferView> gltf_bufferView)
{
    if (!gltf_bufferView->buffer)
    {
        vsg::info("Warning: no buffer available to create BufferView.");
        return {};
    }

    if (!vsg_buffers[gltf_bufferView->buffer.value])
    {
        vsg::info("Warning: no vsg::Data available to create BufferView.");
        return {};
    }

    if (gltf_bufferView->byteStride == 0)
    {
        // The default is 1 and the specification requires 4..252 when the
        // property is present, so this is a malformed document -- but the
        // division below would be by zero, on a number an untrusted server
        // chose.
        vsg::warn("gltf: bufferView declares byteStride 0; it is skipped.");
        return {};
    }

    // TODO: decide whether we need to do anything with the BufferView.target
    auto vsg_buffer = vsg::ubyteArray::create(vsg_buffers[gltf_bufferView->buffer.value],
                                              gltf_bufferView->byteOffset,
                                              gltf_bufferView->byteStride,
                                              gltf_bufferView->byteLength / gltf_bufferView->byteStride);
    return vsg_buffer;
}

namespace
{
    /// KHR_mesh_quantization: a vertex attribute stored as integers.
    ///
    /// POSITION, NORMAL, TANGENT and TEXCOORD_n are normally float, and the
    /// extension lets them be byte or short instead -- which halves or quarters
    /// the file and is what every mesh optimizer emits. The values are brought
    /// back by the node's own scale and translation, which the reader already
    /// applies, so all that is needed here is to widen the components.
    ///
    /// Without this the array reaches the pipeline as a usvec3Array where the
    /// shader set declares a float vec3, and the primitive is dropped: the
    /// model loads, converts, and contains no geometry at all. Six of the 292
    /// models in glTF-Sample-Assets were empty for this reason, Duck and
    /// Lantern among them.
    ///
    /// `normalized` is the accessor's own flag and decides the mapping: the
    /// glTF spec's c/255, c/65535, max(c/127, -1) and max(c/32767, -1). An
    /// un-normalized quantized value is used as-is.
    template<typename SourceArray, typename DestArray, typename DestValue>
    vsg::ref_ptr<vsg::Data> widenComponents(const vsg::ref_ptr<vsg::Data>& in,
                                            bool normalized, double divisor,
                                            bool clampToMinusOne)
    {
        auto source = in.cast<SourceArray>();
        if (!source) return {};

        auto dest = DestArray::create(source->size());
        auto out = dest->begin();
        for (auto& v : *source)
        {
            DestValue& d = *(out++);
            for (std::size_t c = 0; c < d.size(); ++c)
            {
                double value = static_cast<double>(v[c]);
                if (normalized)
                {
                    value /= divisor;
                    if (clampToMinusOne && value < -1.0) value = -1.0;
                }
                d[c] = static_cast<float>(value);
            }
        }
        return dest;
    }

    /// Widen any integer vertex array to float, or return null if it is
    /// already float (or is a type this does not apply to).
    //! Is every index inside the vertex array it addresses?
    //!
    //! Vulkan does not bounds-check an index buffer. An index at or past the
    //! vertex count reads whatever follows the vertex data on the GPU, and the
    //! indices come from the document, so this is the ordinary hostile case
    //! rather than an exotic one -- the same check createPrimitiveOutline
    //! already makes for the outline's own indices.
    //!
    //! Linear in the index count, which is the cheapest it can be: the answer
    //! depends on every element. Measured against the corpus it is not visible
    //! beside the decode that produced the array.
    bool indicesAreInRange(const vsg::ref_ptr<vsg::Data>& indices, uint32_t vertexCount,
                           uint32_t& out_offender)
    {
        if (!indices) return true;

        if (auto u32 = indices.cast<vsg::uintArray>())
        {
            for (auto v : *u32)
                if (v >= vertexCount) { out_offender = v; return false; }
        }
        else if (auto u16 = indices.cast<vsg::ushortArray>())
        {
            for (auto v : *u16)
                if (v >= vertexCount) { out_offender = v; return false; }
        }
        else if (auto u8 = indices.cast<vsg::ubyteArray>())
        {
            for (auto v : *u8)
                if (v >= vertexCount) { out_offender = v; return false; }
        }
        return true;
    }

    //! Give a vertex array the VkFormat its own type implies.
    //!
    //! vsg::Data::Properties::format defaults to VK_FORMAT_UNDEFINED, and
    //! GraphicsPipelineConfigurator::assignArray falls back to the SHADER's
    //! declared format when it finds one:
    //!
    //!     (format != VK_FORMAT_UNDEFINED) ? format : binding.format
    //!
    //! while taking the stride from the array. So an array narrower than the
    //! attribute the shader declares gets the shader's format at the array's
    //! stride, and Vulkan then reads more bytes per vertex than the stride
    //! advances.
    //!
    //! glTF's COLOR_0 may be VEC3 or VEC4 and vsg_Color is declared vec4, so a
    //! VEC3 COLOR_0 was bound as R32G32B32A32_SFLOAT -- 16 bytes -- at a stride
    //! of 12. Every vertex took its alpha from the next vertex's red channel,
    //! and the last vertex read four bytes past the end of the buffer. Nine
    //! documents across the two corpora do this, including both point clouds in
    //! the gallery.
    //!
    //! Setting the format from the array is also what the spec asks for: a
    //! vertex attribute with fewer components than the shader declares has the
    //! missing ones filled with (0, 0, 0, 1), which is exactly the alpha of 1.0
    //! glTF requires for a VEC3 COLOR_0.
    //! Present a base-colour texture as UNORM rather than sRGB.
    //!
    //! WHY, because this looks like the wrong thing to do and is not:
    //!
    //! glTF base colour IS sRGB-encoded, so VK_FORMAT_*_SRGB is the correct
    //! description of the bytes and the sampler is right to linearise them. The
    //! problem is the other end. This renderer's swapchain is UNORM and nothing
    //! encodes back to sRGB on the way out, so a value the sampler correctly
    //! linearised is written to the screen as if it were already display-ready.
    //! Every glTF surface therefore arrives about twice too dark, while the
    //! terrain -- whose imagery is sampled UNORM and passes straight through --
    //! is correct beside it. Measured on the operational tilesets: a roof at
    //! 20/9/5 against a reference of 79/52/40.
    //!
    //! Marking base colour UNORM makes it take the same path as the terrain's
    //! imagery: no linearisation, no re-encode, and the two agree. It is a
    //! compensation, not a correction -- a linear pipeline end to end (sRGB
    //! swapchain, sRGB overlay, terrain shaders encoding properly) is the real
    //! answer, and until then baseColorFactor remains a linear quantity being
    //! multiplied into sRGB-encoded values, so a dark factor still over-darkens.
    //!
    //! Deliberately limited to BASE COLOUR. Normal, metallic-roughness and
    //! occlusion maps are genuinely linear data and are already stored UNORM;
    //! touching them would break what currently works.
    void presentBaseColourAsUnorm(const vsg::ref_ptr<vsg::Data>& image)
    {
        if (!image) return;

        // An escape hatch, because this changes how every glTF in the app
        // looks and "it went darker/brighter" is the kind of claim that should
        // be settled by flipping a switch rather than by two rebuilds.
        static const bool disabled = []{
            const char* v = std::getenv("VSGX_NO_GLTF_UNORM");
            return v && *v && *v != '0';
        }();
        if (disabled) return;

        switch (image->properties.format)
        {
        case VK_FORMAT_R8G8B8A8_SRGB: image->properties.format = VK_FORMAT_R8G8B8A8_UNORM; break;
        case VK_FORMAT_R8G8B8_SRGB:   image->properties.format = VK_FORMAT_R8G8B8_UNORM;   break;
        case VK_FORMAT_R8G8_SRGB:     image->properties.format = VK_FORMAT_R8G8_UNORM;     break;
        case VK_FORMAT_R8_SRGB:       image->properties.format = VK_FORMAT_R8_UNORM;       break;
        case VK_FORMAT_B8G8R8A8_SRGB: image->properties.format = VK_FORMAT_B8G8R8A8_UNORM; break;
        case VK_FORMAT_B8G8R8_SRGB:   image->properties.format = VK_FORMAT_B8G8R8_UNORM;   break;
        default: break;   // already UNORM, compressed, or something else
        }
    }

    //! Put a base-colour FACTOR into the same encoding its texture is in.
    //!
    //! THE OTHER HALF OF presentBaseColourAsUnorm, and exactly the residual
    //! that function's own comment predicted: "baseColorFactor remains a linear
    //! quantity being multiplied into sRGB-encoded values, so a dark factor
    //! still over-darkens."
    //!
    //! glTF defines baseColorFactor as LINEAR. Presenting the texture as UNORM
    //! means its sRGB-encoded bytes pass through un-linearised -- deliberately,
    //! so that glTF matches the terrain beside it. The shader then multiplies
    //! the two, and the operands are in different spaces. The factor has to be
    //! encoded the way the texture already is.
    //!
    //! MEASURED on i3dm_city's roofs, which is what made this findable. The
    //! roof texture is neutral grey (142, 138, 135), so the colour comes
    //! entirely from a factor of [1, 0.798, 0.735]. Multiplied as a linear
    //! quantity that gives (142, 110, 99), a dark brick red. Encoded first it
    //! gives (142, 125, 118), a light warm terracotta -- which is what the
    //! reference renderer shows for the same buildings at the same camera
    //! position.
    //!
    //! THE ERROR IS CHROMATIC rather than a uniform dimming, which is why it
    //! reads as over-saturated as much as too dark, and why it was missed: a
    //! factor of 1.0 is unchanged while 0.735 moves by 19%. The walls of these
    //! buildings looked correct throughout precisely because their factor is
    //! near 1.0 in every channel, so only the roofs gave it away.
    //!
    //! Alpha is not touched: it is coverage, not colour, and has no encoding.
    vsg::vec4 encodeBaseColourFactor(const vsg::vec4& linear)
    {
        // Tied to the same switch as the texture half. The two are one
        // compensation, and disabling half would leave the pipeline less
        // consistent than either end state.
        static const bool disabled = []{
            const char* v = std::getenv("VSGX_NO_GLTF_UNORM");
            return v && *v && *v != '0';
        }();
        if (disabled) return linear;

        auto encode = [](float c) -> float {
            if (c <= 0.0f) return 0.0f;
            if (c >= 1.0f) return 1.0f;      // 1.0 stays 1.0, as it must
            return (c <= 0.0031308f) ? (c * 12.92f)
                                     : (1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f);
        };

        return vsg::vec4(encode(linear.r), encode(linear.g), encode(linear.b),
                         linear.a);
    }

    void setVertexFormatFromType(const vsg::ref_ptr<vsg::Data>& array)
    {
        if (!array || array->properties.format != VK_FORMAT_UNDEFINED) return;

        VkFormat format = VK_FORMAT_UNDEFINED;

        if (array.cast<vsg::floatArray>())        format = VK_FORMAT_R32_SFLOAT;
        else if (array.cast<vsg::vec2Array>())    format = VK_FORMAT_R32G32_SFLOAT;
        else if (array.cast<vsg::vec3Array>())    format = VK_FORMAT_R32G32B32_SFLOAT;
        else if (array.cast<vsg::vec4Array>())    format = VK_FORMAT_R32G32B32A32_SFLOAT;
        else if (array.cast<vsg::ivec4Array>())   format = VK_FORMAT_R32G32B32A32_SINT;
        else if (array.cast<vsg::uivec4Array>())  format = VK_FORMAT_R32G32B32A32_UINT;
        else if (array.cast<vsg::usvec4Array>())  format = VK_FORMAT_R16G16B16A16_UINT;
        else if (array.cast<vsg::ubvec4Array>())  format = VK_FORMAT_R8G8B8A8_UINT;

        // Anything else keeps UNDEFINED and the previous behaviour. Narrowing
        // the fix to the types this reader actually produces for a vertex
        // attribute is deliberate: guessing a format for a type we do not
        // recognise would replace a known-wrong binding with an unknown one.
        if (format != VK_FORMAT_UNDEFINED) array->properties.format = format;
    }

    vsg::ref_ptr<vsg::Data> widenQuantizedAttribute(const vsg::ref_ptr<vsg::Data>& array,
                                                    bool normalized)
    {
        if (!array) return {};

        // Already float: nothing to do, and saying so by returning null keeps
        // the caller's "did I change it" test to one check.
        if (array.cast<vsg::vec2Array>() || array.cast<vsg::vec3Array>() ||
            array.cast<vsg::vec4Array>())
        {
            return {};
        }

        vsg::ref_ptr<vsg::Data> result;

        // unsigned byte -> /255,  byte -> /127 clamped, unsigned short -> /65535,
        // short -> /32767 clamped.  glTF 2.0, "Animation Sampler" table and
        // KHR_mesh_quantization.
        if ((result = widenComponents<vsg::ubvec2Array, vsg::vec2Array, vsg::vec2>(array, normalized, 255.0, false))) return result;
        if ((result = widenComponents<vsg::ubvec3Array, vsg::vec3Array, vsg::vec3>(array, normalized, 255.0, false))) return result;
        if ((result = widenComponents<vsg::ubvec4Array, vsg::vec4Array, vsg::vec4>(array, normalized, 255.0, false))) return result;

        if ((result = widenComponents<vsg::bvec2Array, vsg::vec2Array, vsg::vec2>(array, normalized, 127.0, true))) return result;
        if ((result = widenComponents<vsg::bvec3Array, vsg::vec3Array, vsg::vec3>(array, normalized, 127.0, true))) return result;
        if ((result = widenComponents<vsg::bvec4Array, vsg::vec4Array, vsg::vec4>(array, normalized, 127.0, true))) return result;

        if ((result = widenComponents<vsg::usvec2Array, vsg::vec2Array, vsg::vec2>(array, normalized, 65535.0, false))) return result;
        if ((result = widenComponents<vsg::usvec3Array, vsg::vec3Array, vsg::vec3>(array, normalized, 65535.0, false))) return result;
        if ((result = widenComponents<vsg::usvec4Array, vsg::vec4Array, vsg::vec4>(array, normalized, 65535.0, false))) return result;

        if ((result = widenComponents<vsg::svec2Array, vsg::vec2Array, vsg::vec2>(array, normalized, 32767.0, true))) return result;
        if ((result = widenComponents<vsg::svec3Array, vsg::vec3Array, vsg::vec3>(array, normalized, 32767.0, true))) return result;
        if ((result = widenComponents<vsg::svec4Array, vsg::vec4Array, vsg::vec4>(array, normalized, 32767.0, true))) return result;

        return {};
    }
}

vsg::ref_ptr<vsg::Data> gltf::SceneGraphBuilder::createArray(const std::string& type, uint32_t componentType, glTFid bufferView, uint32_t offset, uint32_t count)
{
    vsg::ref_ptr<vsg::Data> vsg_data;

    auto vsg_bufferView = vsg_bufferViews[bufferView.value];

    auto stride = [&](uint32_t s) -> uint32_t {
        return std::max(vsg_bufferView->properties.stride, s);
    };

    switch (componentType)
    {
    case (COMPONENT_TYPE_BYTE):
        if (type == "SCALAR")
            vsg_data = vsg::byteArray::create(vsg_bufferView, offset, stride(1), count);
        else if (type == "VEC2")
            vsg_data = vsg::bvec2Array::create(vsg_bufferView, offset, stride(2), count);
        else if (type == "VEC3")
            vsg_data = vsg::bvec3Array::create(vsg_bufferView, offset, stride(3), count);
        else if (type == "VEC4")
            vsg_data = vsg::bvec4Array::create(vsg_bufferView, offset, stride(4), count);
        else
            vsg::warn("Unsupported componentType = ", componentType);
        break;
    case (COMPONENT_TYPE_UNSIGNED_BYTE):
        if (type == "SCALAR")
            vsg_data = vsg::ubyteArray::create(vsg_bufferView, offset, stride(1), count);
        else if (type == "VEC2")
            vsg_data = vsg::ubvec2Array::create(vsg_bufferView, offset, stride(2), count);
        else if (type == "VEC3")
            vsg_data = vsg::ubvec3Array::create(vsg_bufferView, offset, stride(3), count);
        else if (type == "VEC4")
            vsg_data = vsg::ubvec4Array::create(vsg_bufferView, offset, stride(4), count);
        else
            vsg::warn("Unsupported componentType = ", componentType);
        break;
    case (COMPONENT_TYPE_SHORT):
        if (type == "SCALAR")
            vsg_data = vsg::shortArray::create(vsg_bufferView, offset, stride(2), count);
        else if (type == "VEC2")
            vsg_data = vsg::svec2Array::create(vsg_bufferView, offset, stride(4), count);
        else if (type == "VEC3")
            vsg_data = vsg::svec3Array::create(vsg_bufferView, offset, stride(6), count);
        else if (type == "VEC4")
            vsg_data = vsg::svec4Array::create(vsg_bufferView, offset, stride(8), count);
        else
            vsg::warn("Unsupported componentType = ", componentType);
        break;
    case (COMPONENT_TYPE_UNSIGNED_SHORT):
        if (type == "SCALAR")
            vsg_data = vsg::ushortArray::create(vsg_bufferView, offset, stride(2), count);
        else if (type == "VEC2")
            vsg_data = vsg::usvec2Array::create(vsg_bufferView, offset, stride(4), count);
        else if (type == "VEC3")
            vsg_data = vsg::usvec3Array::create(vsg_bufferView, offset, stride(6), count);
        else if (type == "VEC4")
            vsg_data = vsg::usvec4Array::create(vsg_bufferView, offset, stride(8), count);
        else
            vsg::warn("Unsupported componentType = ", componentType);
        break;
    case (COMPONENT_TYPE_INT):
        if (type == "SCALAR")
            vsg_data = vsg::intArray::create(vsg_bufferView, offset, stride(4), count);
        else if (type == "VEC2")
            vsg_data = vsg::ivec2Array::create(vsg_bufferView, offset, stride(8), count);
        else if (type == "VEC3")
            vsg_data = vsg::ivec3Array::create(vsg_bufferView, offset, stride(12), count);
        else if (type == "VEC4")
            vsg_data = vsg::ivec4Array::create(vsg_bufferView, offset, stride(16), count);
        else
            vsg::warn("Unsupported componentType = ", componentType);
        break;
    case (COMPONENT_TYPE_UNSIGNED_INT):
        if (type == "SCALAR")
            vsg_data = vsg::uintArray::create(vsg_bufferView, offset, stride(4), count);
        else if (type == "VEC2")
            vsg_data = vsg::uivec2Array::create(vsg_bufferView, offset, stride(8), count);
        else if (type == "VEC3")
            vsg_data = vsg::uivec3Array::create(vsg_bufferView, offset, stride(12), count);
        else if (type == "VEC4")
            vsg_data = vsg::uivec4Array::create(vsg_bufferView, offset, stride(16), count);
        else
            vsg::warn("Unsupported componentType = ", componentType);
        break;
    case (COMPONENT_TYPE_FLOAT):
        if (type == "SCALAR")
            vsg_data = vsg::floatArray::create(vsg_bufferView, offset, stride(4), count);
        else if (type == "VEC2")
            vsg_data = vsg::vec2Array::create(vsg_bufferView, offset, stride(8), count);
        else if (type == "VEC3")
            vsg_data = vsg::vec3Array::create(vsg_bufferView, offset, stride(12), count);
        else if (type == "VEC4")
            vsg_data = vsg::vec4Array::create(vsg_bufferView, offset, stride(16), count);
        //else if (type=="MAT2")   vsg_data = vsg::mat2Array::create(vsg_bufferView, offset, stride(16), count);
        //else if (type=="MAT3")   vsg_data = vsg::mat3Array::create(vsg_bufferView, offset, stride(36), count);
        else if (type == "MAT4")
            vsg_data = vsg::mat4Array::create(vsg_bufferView, offset, stride(64), count);
        else
            vsg::warn("Unsupported componentType = ", componentType);
        break;
    case (COMPONENT_TYPE_DOUBLE):
        if (type == "SCALAR")
            vsg_data = vsg::doubleArray::create(vsg_bufferView, offset, stride(8), count);
        else if (type == "VEC2")
            vsg_data = vsg::dvec2Array::create(vsg_bufferView, offset, stride(16), count);
        else if (type == "VEC3")
            vsg_data = vsg::dvec3Array::create(vsg_bufferView, offset, stride(24), count);
        else if (type == "VEC4")
            vsg_data = vsg::dvec4Array::create(vsg_bufferView, offset, stride(32), count);
        //else if (type=="MAT2")   vsg_data = vsg::dmat2Array::create(vsg_bufferView, offset, stride(32), count);
        //else if (type=="MAT3")   vsg_data = vsg::dmat3Array::create(vsg_bufferView, offset, stride(72), count);
        else if (type == "MAT4")
            vsg_data = vsg::dmat4Array::create(vsg_bufferView, offset, stride(128), count);
        else
            vsg::warn("Unsupported componentType = ", componentType);
        break;
    }

    if (cloneAccessors)
    {
        vsg::info("clonning vsg_data ", vsg_data);
        vsg_data = vsg::clone(vsg_data);
        vsg::info("clonned vsg_data ", vsg_data);
    }

    return vsg_data;
}

vsg::ref_ptr<vsg::Data> gltf::SceneGraphBuilder::createAccessor(vsg::ref_ptr<gltf::Accessor> gltf_accessor)
{
    if (!gltf_accessor->bufferView)
    {
        vsg::info("Warning: no bufferView available to create Accessor.");
        return {};
    }

    if (!vsg_bufferViews[gltf_accessor->bufferView.value])
    {
        vsg::info("Warning: no vsg::Data available to create BufferView.");
        return {};
    }

    auto vsg_data = createArray(gltf_accessor->type, gltf_accessor->componentType, gltf_accessor->bufferView, gltf_accessor->byteOffset, gltf_accessor->count);

    if (gltf_accessor->sparse)
    {
        auto sparse = gltf_accessor->sparse;
        auto vsg_indices = createArray("SCALAR", sparse->indices->componentType, sparse->indices->bufferView, sparse->indices->byteOffset, sparse->count);
        auto vsg_values = createArray(gltf_accessor->type, gltf_accessor->componentType, sparse->values->bufferView, sparse->values->byteOffset, sparse->count);

        if (auto uint_indices = vsg_indices.cast<vsg::uintArray>())
        {
            for (size_t i = 0; i < uint_indices->size(); ++i)
            {
                std::memcpy(vsg_data->dataPointer(uint_indices->at(i)), vsg_values->dataPointer(i), vsg_data->valueSize());
            }
        }
        else if (auto ushort_indices = vsg_indices.cast<vsg::ushortArray>())
        {
            for (size_t i = 0; i < ushort_indices->size(); ++i)
            {
                std::memcpy(vsg_data->dataPointer(ushort_indices->at(i)), vsg_values->dataPointer(i), vsg_data->valueSize());
            }
        }
        else
        {
            vsg::warn("gltf::SceneGraphBuilder::createAccessor(...) sparse indices type (", sparse->indices->componentType, " not supported. ");
        }
    }

    return vsg_data;
}

vsg::ref_ptr<vsg::Camera> gltf::SceneGraphBuilder::createCamera(vsg::ref_ptr<gltf::Camera> gltf_camera)
{
    auto vsg_camera = vsg::Camera::create();

    if (gltf_camera->perspective)
    {
        auto perspective = gltf_camera->perspective;
        // note, vsg::Perspective implements GLU Perspective style settings so uses degress for fov, while glTF uses radians so need to convert to degrees
        vsg_camera->projectionMatrix = vsg::Perspective::create(vsg::degrees(perspective->yfov), perspective->aspectRatio, perspective->znear, perspective->zfar);
    }

    if (gltf_camera->orthographic)
    {
        auto orthographic = gltf_camera->orthographic;
        double halfWidth = orthographic->xmag;  // TODO: figure how to map to GLU/VSG style orthographic
        double halfHeight = orthographic->ymag; // TODO: figure how to mapto GLU/VSG style orthographic
        vsg_camera->projectionMatrix = vsg::Orthographic::create(-halfWidth, halfWidth, -halfHeight, halfHeight, orthographic->znear, orthographic->zfar);
    }

    vsg_camera->name = gltf_camera->name;
    assign_extras(*gltf_camera, *vsg_camera);

    return vsg_camera;
}

vsg::ref_ptr<vsg::Data> gltf::SceneGraphBuilder::createImage(vsg::ref_ptr<gltf::Image> gltf_image)
{
    if (gltf_image->data)
    {
        // vsg::info("createImage(", gltf_image, ") gltf_image->data = ", gltf_image->data);
        return gltf_image->data;
    }
    else if (gltf_image->bufferView)
    {
        auto data = vsg_bufferViews[gltf_image->bufferView.value];
        // vsg::info("createImage(", gltf_image, ") bufferView = ", gltf_image->bufferView, ", vsg_bufferView = ", data);
        return data;
    }
    else
    {
        vsg::info("createImage(", gltf_image, ") uri = ", gltf_image->uri, ", nothing to create vsg::Data image from.");
        return {};
    }
}

vsg::ref_ptr<vsg::Sampler> gltf::SceneGraphBuilder::createSampler(vsg::ref_ptr<gltf::Sampler> gltf_sampler)
{
    auto vsg_sampler = vsg::Sampler::create();

    vsg_sampler->maxAnisotropy = maxAnisotropy;
    vsg_sampler->anisotropyEnable = (maxAnisotropy > 0.0f) ? VK_TRUE : VK_FALSE;

    // assume mipmapping.
    vsg_sampler->minLod = 0.0f;
    vsg_sampler->maxLod = 16.0f;

    // See https://docs.vulkan.org/spec/latest/chapters/samplers.html for suggestions on mapping from OpenGL style to Vulkan
    switch (gltf_sampler->minFilter)
    {
    case (9728): // NEAREST
        vsg_sampler->minFilter = VK_FILTER_NEAREST;
        vsg_sampler->mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        vsg_sampler->minLod = 0.0f;
        vsg_sampler->maxLod = 0.25f;
        break;
    case (9729): // LINEAR
        vsg_sampler->minFilter = VK_FILTER_LINEAR;
        vsg_sampler->mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        vsg_sampler->minLod = 0.0f;
        vsg_sampler->maxLod = 0.25f;
        break;
    case (9984): // NEAREST_MIPMAP_NEAREST
        vsg_sampler->minFilter = VK_FILTER_NEAREST;
        vsg_sampler->mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        break;
    case (9985): // LINEAR_MIPMAP_NEAREST
        vsg_sampler->minFilter = VK_FILTER_LINEAR;
        vsg_sampler->mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        break;
    case (9986): // NEAREST_MIPMAP_LINEAR
        vsg_sampler->minFilter = VK_FILTER_NEAREST;
        vsg_sampler->mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        break;
    case (9987): // LINEAR_MIPMAP_LINEAR
        vsg_sampler->minFilter = VK_FILTER_LINEAR;
        vsg_sampler->mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        break;
    default:
        vsg::debug("gltf_sampler->minFilter value of ", gltf_sampler->minFilter, " not set, using linear mipmap linear.");
        vsg_sampler->minFilter = VK_FILTER_LINEAR;
        vsg_sampler->mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        break;
    }

    switch (gltf_sampler->magFilter)
    {
    case (9728):
        vsg_sampler->magFilter = VK_FILTER_NEAREST;
        break;
    case (9729):
        vsg_sampler->magFilter = VK_FILTER_LINEAR;
        break;
    default:
        vsg::debug("gltf_sampler->magFilter value of ", gltf_sampler->magFilter, " not set, using default of linear.");
        vsg_sampler->magFilter = VK_FILTER_LINEAR;
        break;
    }

    auto addressMode = [](uint32_t wrap) -> VkSamplerAddressMode {
        switch (wrap)
        {
        case (33071): // CLAMP_TO_EDGE
            return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        case (33648): // MIRRORED_REPEAT
            return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        case (10497): // REPEAT
            return VK_SAMPLER_ADDRESS_MODE_REPEAT;
        default:
            vsg::warn("gltf_sampler->wrap* value of ", wrap, " not supported.");
            break;
        }
        return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    };

    vsg_sampler->addressModeU = addressMode(gltf_sampler->wrapS);
    vsg_sampler->addressModeV = addressMode(gltf_sampler->wrapT);

    if (sharedObjects)
    {
        sharedObjects->share(vsg_sampler);
    }

    // vsg::info("created sampler { ", vsg_sampler->minFilter, ", ", vsg_sampler->magFilter, ", ", vsg_sampler->mipmapMode, ", ",  gltf_sampler->wrapS, ", ", gltf_sampler->wrapT, "}");

    return vsg_sampler;
}

gltf::SceneGraphBuilder::SamplerImage gltf::SceneGraphBuilder::createTexture(vsg::ref_ptr<gltf::Texture> gltf_texture)
{
    SamplerImage samplerImage;

    if (gltf_texture->sampler)
    {
        samplerImage.sampler = vsg_samplers[gltf_texture->sampler.value];
    }

    if (gltf_texture->source)
    {
        samplerImage.image = vsg_images[gltf_texture->source.value];
    }

    return samplerImage;
}

vsg::ref_ptr<vsg::DescriptorConfigurator> gltf::SceneGraphBuilder::createPbrMaterial(vsg::ref_ptr<gltf::Material> gltf_material)
{
    auto vsg_material = vsg::DescriptorConfigurator::create();

    vsg_material->shaderSet = getOrCreatePbrShaderSet();

    vsg_material->two_sided = gltf_material->doubleSided;
    if (vsg_material->two_sided) vsg_material->defines.insert("VSG_TWO_SIDED_LIGHTING");

    auto pbrMaterialValue = vsg::PbrMaterialValue::create();
    auto& pbrMaterial = pbrMaterialValue->value();

    auto texCoordIndicesValue = vsg::TexCoordIndicesValue::create();
    auto& texCoordIndices = texCoordIndicesValue->value();

    if (gltf_material->pbrMetallicRoughness.baseColorFactor.values.size() == 4)
    {
        auto& baseColorFactor = gltf_material->pbrMetallicRoughness.baseColorFactor.values;
        pbrMaterial.baseColorFactor = encodeBaseColourFactor(
            vsg::vec4(baseColorFactor[0], baseColorFactor[1],
                      baseColorFactor[2], baseColorFactor[3]));
        // vsg::info("Assigned baseColorFacator ", pbrMaterial.baseColorFactor);
    }

    if (gltf_material->pbrMetallicRoughness.baseColorTexture.index)
    {
        auto& textureInfo = gltf_material->pbrMetallicRoughness.baseColorTexture;
        auto& texture = vsg_textures[textureInfo.index.value];
        if (texture.image)
        {
            // vsg::info("Assigned diffuseMap ", texture.image, ", ", texture.sampler);
            presentBaseColourAsUnorm(texture.image);
            vsg_material->assignTexture("diffuseMap", texture.image, texture.sampler);
            texCoordIndices.diffuseMap = textureInfo.texCoord;

            if (auto texture_transform = textureInfo.extension<KHR_texture_transform>("KHR_texture_transform"))
            {
                vsg_material->setObject("KHR_texture_transform", texture_transform);
            }
        }
        else
        {
            vsg::warn("Could not assign diffuseMap ", textureInfo.index);
        }
    }

    pbrMaterial.metallicFactor = gltf_material->pbrMetallicRoughness.metallicFactor;
    pbrMaterial.roughnessFactor = gltf_material->pbrMetallicRoughness.roughnessFactor;

    if (gltf_material->pbrMetallicRoughness.metallicRoughnessTexture.index)
    {
        auto& textureInfo = gltf_material->pbrMetallicRoughness.metallicRoughnessTexture;
        auto& texture = vsg_textures[textureInfo.index.value];
        if (texture.image)
        {
            // vsg::info("Assigned metallicRoughnessTexture ", texture.image, ", ", texture.sampler);
            vsg_material->assignTexture("mrMap", texture.image, texture.sampler);
            texCoordIndices.mrMap = textureInfo.texCoord;
        }
        else
        {
            vsg::warn("Could not assign metallicRoughnessTexture ", textureInfo.index);
        }
    }

    // TODO : pbrMaterial.diffuseFactor? No glTF mapping?

    if (gltf_material->normalTexture.index)
    {
        // TODO: gltf_material->normalTexture.scale

        auto& textureInfo = gltf_material->normalTexture;
        auto& texture = vsg_textures[textureInfo.index.value];
        if (texture.image)
        {
            // vsg::info("Assigned normalTexture ", texture.image, ", ", texture.sampler, ", scale = ", gltf_material->normalTexture.scale);
            vsg_material->assignTexture("normalMap", texture.image, texture.sampler);
            texCoordIndices.normalMap = textureInfo.texCoord;
        }
        else
        {
            vsg::warn("Could not assign normalTexture ", textureInfo.index);
        }
    }

    if (gltf_material->occlusionTexture.index)
    {
        // TODO: gltf_material->occlusionTexture.strength

        auto& textureInfo = gltf_material->occlusionTexture;
        auto& texture = vsg_textures[textureInfo.index.value];
        if (texture.image)
        {
            // vsg::info("Assigned occlusionTexture ", texture.image, ", ", texture.sampler, ", strength = ", textureInfo.strength);
            vsg_material->assignTexture("aoMap", texture.image, texture.sampler);
            texCoordIndices.aoMap = textureInfo.texCoord;
        }
        else
        {
            vsg::warn("Could not assign occlusionTexture ", textureInfo.index);
        }
    }

    if (gltf_material->emissiveTexture.index)
    {
        auto& textureInfo = gltf_material->emissiveTexture;
        auto& texture = vsg_textures[textureInfo.index.value];
        if (texture.image)
        {
            // vsg::info("Assigned emissiveTexture ", texture.image, ", ", texture.sampler);
            vsg_material->assignTexture("emissiveMap", texture.image, texture.sampler);
            texCoordIndices.emissiveMap = textureInfo.texCoord;
        }
        else
        {
            vsg::warn("Could not assign emissiveTexture ", textureInfo.index);
        }
    }

    if (gltf_material->emissiveFactor.values.size() >= 3)
    {
        pbrMaterial.emissiveFactor.set(gltf_material->emissiveFactor.values[0], gltf_material->emissiveFactor.values[1], gltf_material->emissiveFactor.values[2], 1.0);
        // vsg::info("Set pbrMaterial.emissiveFactor = ", pbrMaterial.emissiveFactor);
    }

    if (gltf_material->alphaMode == "BLEND")
    {
        vsg_material->blending = true;
    }
    else if (gltf_material->alphaMode == "MASK")
    {
        // vsg_material->blending = true; // TODO, do we need to enable blending?
        vsg_material->defines.insert("VSG_ALPHA_TEST");
        pbrMaterial.alphaMaskCutoff = gltf_material->alphaCutoff;
    }

    if (auto materials_specular = gltf_material->extension<KHR_materials_specular>("KHR_materials_specular"))
    {
        float sf = materials_specular->specularFactor;
        pbrMaterial.specularFactor.set(sf, sf, sf, 1.0);

        if (materials_specular->specularTexture.index)
        {
            auto& texture = vsg_textures[materials_specular->specularTexture.index.value];
            if (texture.image)
            {
                // vsg::info("Assigned specularTexture ", texture.image, ", ", texture.sampler);
                vsg_material->assignTexture("specularMap", texture.image, texture.sampler);
            }
            else
            {
                vsg::warn("Could not assign specularTexture ", materials_specular->specularTexture.index);
            }
        }

        if (materials_specular->specularColorFactor.values.size() >= 3)
        {
            auto& specularColorFactor = materials_specular->specularColorFactor.values;
            pbrMaterial.specularFactor.set(specularColorFactor[0], specularColorFactor[1], specularColorFactor[2], 1.0); // TODO, alpha value? Shoult it be specularFactor?
            // vsg::info("Assigned specularColorFactor pbrMaterial.specularFactor ", pbrMaterial.specularFactor);
        }

        if (materials_specular->specularColorTexture.index)
        {
            vsg::info("Not assigned yet: specularColorTexture = ", materials_specular->specularColorTexture.index, ", ", materials_specular->specularColorTexture.texCoord);
        }
    }

    if (auto materials_pbrSpecularGlossiness = gltf_material->extension<KHR_materials_pbrSpecularGlossiness>("KHR_materials_pbrSpecularGlossiness"))
    {

        if (materials_pbrSpecularGlossiness->diffuseFactor.values.size() >= 3)
        {
            auto& diffuseFactor = materials_pbrSpecularGlossiness->diffuseFactor.values;
            pbrMaterial.diffuseFactor.set(diffuseFactor[0], diffuseFactor[1], diffuseFactor[2], 1.0);
        }

        if (materials_pbrSpecularGlossiness->specularFactor.values.size() >= 3)
        {
            auto& specularFactor = materials_pbrSpecularGlossiness->specularFactor.values;
            pbrMaterial.specularFactor.set(specularFactor[0], specularFactor[1], specularFactor[2], 1.0);
        }

        if (materials_pbrSpecularGlossiness->diffuseTexture.index)
        {
            auto& texture = vsg_textures[materials_pbrSpecularGlossiness->diffuseTexture.index.value];
            if (texture.image)
            {
                vsg_material->assignTexture("diffuseMap", texture.image, texture.sampler);
            }
            else
            {
                vsg::warn("Could not assign diffuseTexture ", materials_pbrSpecularGlossiness->diffuseTexture.index);
            }
        }

        if (materials_pbrSpecularGlossiness->specularGlossinessTexture.index)
        {
            auto& texture = vsg_textures[materials_pbrSpecularGlossiness->specularGlossinessTexture.index.value];
            if (texture.image)
            {
                vsg_material->assignTexture("specularMap", texture.image, texture.sampler);
            }
            else
            {
                vsg::warn("Could not assign specularTexture ", materials_pbrSpecularGlossiness->specularGlossinessTexture.index);
            }
        }

        pbrMaterial.specularFactor.a = materials_pbrSpecularGlossiness->glossinessFactor;

        vsg_material->defines.insert("VSG_WORKFLOW_SPECGLOSS");
    }

    if (auto materials_emissive_strength = gltf_material->extension<KHR_materials_emissive_strength>("KHR_materials_emissive_strength"))
    {
        pbrMaterial.emissiveFactor.a = materials_emissive_strength->emissiveStrength;
    }

#if 0
    if (auto materials_ior = gltf_material->extension<KHR_materials_ior>("KHR_materials_ior"))
    {
        vsg::info("Have Index Of Refraction: ", materials_ior);
    }

    if (gltf_material->extensions)
    {
        for(auto& [name, schema] : gltf_material->extensions->values)
        {
            vsg::info("extensions ", name, ", ", schema);
        }
    }
#endif

    vsg_material->assignDescriptor("material", pbrMaterialValue);
    vsg_material->assignDescriptor("texCoordIndices", texCoordIndicesValue);

    // TODO: vsg_material->defines.insert("VSG_WORKFLOW_SPECGLOSS");
    // TODO: VSG -> detailMap

    return vsg_material;
}

vsg::ref_ptr<vsg::DescriptorConfigurator> gltf::SceneGraphBuilder::createUnlitMaterial(vsg::ref_ptr<gltf::Material> gltf_material)
{
    auto vsg_material = vsg::DescriptorConfigurator::create();

    vsg_material->shaderSet = getOrCreateFlatShaderSet();

    auto phongMaterialValue = vsg::PhongMaterialValue::create();
    auto& phongMaterial = phongMaterialValue->value();

    auto texCoordIndicesValue = vsg::TexCoordIndicesValue::create();
    auto& texCoordIndices = texCoordIndicesValue->value();

    if (gltf_material->pbrMetallicRoughness.baseColorFactor.values.size() == 4)
    {
        auto& baseColorFactor = gltf_material->pbrMetallicRoughness.baseColorFactor.values;
        phongMaterial.diffuse = encodeBaseColourFactor(
            vsg::vec4(baseColorFactor[0], baseColorFactor[1],
                      baseColorFactor[2], baseColorFactor[3]));
        // vsg::info("Assigned phongMaterial.diffuse ", pbrMaterial.baseColorFactor);
    }

    if (gltf_material->pbrMetallicRoughness.baseColorTexture.index)
    {
        auto& textureInfo = gltf_material->pbrMetallicRoughness.baseColorTexture;
        auto& texture = vsg_textures[textureInfo.index.value];
        if (texture.image)
        {
            // vsg::info("Assigned diffuseMap ", texture.image, ", ", texture.sampler);
            presentBaseColourAsUnorm(texture.image);
            vsg_material->assignTexture("diffuseMap", texture.image, texture.sampler);
            texCoordIndices.diffuseMap = textureInfo.texCoord;

            if (auto texture_transform = textureInfo.extension<KHR_texture_transform>("KHR_texture_transform"))
            {
                vsg_material->setObject("KHR_texture_transform", texture_transform);
            }
        }
        else
        {
            vsg::warn("Could not assign diffuseMap ", gltf_material->pbrMetallicRoughness.baseColorTexture.index);
        }
    }

    if (gltf_material->alphaMode == "BLEND")
    {
        vsg_material->blending = true;
    }
    else if (gltf_material->alphaMode == "MASK")
    {
        // vsg_material->blending = true; // TODO, do we need to enable blending?
        vsg_material->defines.insert("VSG_ALPHA_TEST");
        phongMaterial.alphaMaskCutoff = gltf_material->alphaCutoff;
    }

    vsg_material->assignDescriptor("material", phongMaterialValue);

    return vsg_material;
}

vsg::ref_ptr<vsg::DescriptorConfigurator> gltf::SceneGraphBuilder::createMaterial(vsg::ref_ptr<gltf::Material> gltf_material)
{
    auto vsg_material = vsg::DescriptorConfigurator::create();
    if (auto materials_unlit = gltf_material->extension<KHR_materials_unlit>("KHR_materials_unlit"))
        return createUnlitMaterial(gltf_material);
    else
        return createPbrMaterial(gltf_material);
}

// ---------------------------------------------------------------------------
// 3D Tiles 1.1 per-feature metadata: EXT_structural_metadata + EXT_mesh_features
//
// 1.1 says what a 1.0 batch table said, in a different place and with a real
// type system. A document-level `schema` names classes and the type of each of
// their properties; a `propertyTable` holds one column per property, in a
// bufferView; and a primitive's EXT_mesh_features says which of its
// `_FEATURE_ID_n` attributes indexes which table.
//
// Neither extension has a registered schema in this reader, so both arrive as
// the generic metadata tree JSONtoMetaDataSchema builds for anything
// unrecognised: nested objects are vsg::Object, arrays are vsg::Objects,
// numbers are doubleValue and strings are stringValue. That is enough to read
// them without writing a second SAX parser for a structure this reader only
// ever needs to walk, never to round-trip.
//
// Both spellings end up in the same Tiles3D::FeatureTable, so nothing
// downstream of the reader has to know which one a tileset used.
// ---------------------------------------------------------------------------
namespace
{
    vsg::Object* metaObject(vsg::Object& o, const char* name)
    {
        return o.getObject(name);
    }

    vsg::Objects* metaArray(vsg::Object& o, const char* name)
    {
        return dynamic_cast<vsg::Objects*>(o.getObject(name));
    }

    //! A number from the metadata tree. False when absent or not a number, so a
    //! missing index can never be mistaken for index 0.
    bool metaNumber(vsg::Object& o, const char* name, double& out)
    {
        if (auto v = dynamic_cast<vsg::doubleValue*>(o.getObject(name)))
        {
            out = v->value();
            return true;
        }
        return false;
    }

    std::string metaString(vsg::Object& o, const char* name)
    {
        if (auto v = dynamic_cast<vsg::stringValue*>(o.getObject(name)))
            return v->value();
        return {};
    }

    //! EXT_structural_metadata spells component types by name; the rest of glTF
    //! uses the GL constants. Returns 0 for the two 64-bit integer types, which
    //! have no glTF constant and no vsg::Array to hold them.
    uint32_t componentTypeConstant(const std::string& name)
    {
        if (name == "INT8") return gltf::COMPONENT_TYPE_BYTE;
        if (name == "UINT8") return gltf::COMPONENT_TYPE_UNSIGNED_BYTE;
        if (name == "INT16") return gltf::COMPONENT_TYPE_SHORT;
        if (name == "UINT16") return gltf::COMPONENT_TYPE_UNSIGNED_SHORT;
        if (name == "INT32") return gltf::COMPONENT_TYPE_INT;
        if (name == "UINT32") return gltf::COMPONENT_TYPE_UNSIGNED_INT;
        if (name == "FLOAT32") return gltf::COMPONENT_TYPE_FLOAT;
        if (name == "FLOAT64") return gltf::COMPONENT_TYPE_DOUBLE;
        return gltf::COMPONENT_TYPE_UNDEFINED;
    }

    //! A STRING column: UTF-8 bytes in one bufferView, and count+1 offsets into
    //! them in another.
    vsg::ref_ptr<vsg::Data> decodeStringColumn(gltf::SceneGraphBuilder& builder,
                                               uint32_t valuesView, uint32_t offsetsView,
                                               const std::string& offsetType, uint32_t count)
    {
        if (valuesView >= builder.vsg_bufferViews.size()) return {};
        if (offsetsView >= builder.vsg_bufferViews.size()) return {};

        auto values = builder.vsg_bufferViews[valuesView];
        auto offsets = builder.vsg_bufferViews[offsetsView];
        if (!values || !offsets) return {};

        // Every offset is read, so the width has to be right rather than
        // assumed: a UINT32 default read as UINT16 silently halves the string.
        const uint32_t width =
            (offsetType == "UINT8") ? 1u :
            (offsetType == "UINT16") ? 2u :
            (offsetType == "UINT64") ? 8u : 4u;      // UINT32 is the default

        const uint8_t* offsetBytes = static_cast<const uint8_t*>(offsets->dataPointer());
        const size_t offsetsSize = offsets->dataSize();

        // count + 1 offsets: the last one closes the last string.
        if (offsetsSize < size_t(count + 1) * width) return {};

        auto readOffset = [&](uint32_t i) -> uint64_t {
            const uint8_t* p = offsetBytes + size_t(i) * width;
            uint64_t v = 0;
            std::memcpy(&v, p, width);               // little-endian, as glTF is
            return v;
        };

        const uint8_t* text = static_cast<const uint8_t*>(values->dataPointer());
        const size_t textSize = values->dataSize();

        auto strings = vsg::stringArray::create(count);
        for (uint32_t i = 0; i < count; ++i)
        {
            const uint64_t begin = readOffset(i);
            const uint64_t end = readOffset(i + 1);

            // Offsets come from the document. Anything that would read outside
            // the values view, or backwards, yields an empty string rather than
            // a read off the end of the buffer.
            if (end < begin || end > textSize)
            {
                strings->at(i) = std::string();
                continue;
            }
            strings->at(i) = std::string(reinterpret_cast<const char*>(text + begin),
                                         size_t(end - begin));
        }
        return strings;
    }

    //! One property column, decoded by the type its class declares.
    vsg::ref_ptr<vsg::Data> createPropertyColumn(gltf::SceneGraphBuilder& builder,
                                                 vsg::Object& classProperty,
                                                 vsg::Object& tableProperty,
                                                 uint32_t count,
                                                 const std::string& name)
    {
        double valuesView = 0.0;
        if (!metaNumber(tableProperty, "values", valuesView)) return {};
        if (valuesView < 0.0 || valuesView >= double(builder.vsg_bufferViews.size())) return {};

        const std::string type = metaString(classProperty, "type");

        // Variable-length arrays need arrayOffsets and a second indirection.
        // Refused rather than half-read: a column silently truncated to its
        // first element would read as data, not as an omission.
        double unusedArrayOffsets = 0.0;
        if (metaNumber(tableProperty, "arrayOffsets", unusedArrayOffsets))
        {
            vsg::warn("gltf: EXT_structural_metadata property \"", name,
                      "\" is a variable-length array; it is not read.");
            return {};
        }

        if (type == "STRING")
        {
            double offsetsView = 0.0;
            if (!metaNumber(tableProperty, "stringOffsets", offsetsView)) return {};
            if (offsetsView < 0.0 || offsetsView >= double(builder.vsg_bufferViews.size()))
                return {};

            return decodeStringColumn(builder, uint32_t(valuesView), uint32_t(offsetsView),
                                      metaString(tableProperty, "stringOffsetType"), count);
        }

        // BOOLEAN is a bitstream and ENUM needs the schema's enum tables. Both
        // are legal and neither is read here; saying so beats an empty column.
        if (type == "BOOLEAN" || type == "ENUM")
        {
            vsg::warn("gltf: EXT_structural_metadata property \"", name, "\" has type ",
                      type, ", which is not read.");
            return {};
        }

        const uint32_t componentType =
            componentTypeConstant(metaString(classProperty, "componentType"));

        if (componentType == gltf::COMPONENT_TYPE_UNDEFINED)
        {
            vsg::warn("gltf: EXT_structural_metadata property \"", name,
                      "\" has component type ", metaString(classProperty, "componentType"),
                      ", which has no array type here; it is dropped.");
            return {};
        }

        // createArray dereferences the view without checking it. A view can be
        // null -- createBufferView returns null for a buffer that did not load,
        // and now for a byteStride of 0 -- and this is the first caller that can
        // reach one, because a property table names bufferViews DIRECTLY rather
        // than through an accessor the reader has already validated.
        if (!builder.vsg_bufferViews[size_t(valuesView)])
        {
            vsg::warn("gltf: EXT_structural_metadata property \"", name,
                      "\" names bufferView ", uint32_t(valuesView),
                      ", which did not load; the property is dropped.");
            return {};
        }

        return builder.createArray(type, componentType, gltf::glTFid{uint32_t(valuesView)},
                                   0, count);
    }

    //! The property table `index` of EXT_structural_metadata, as a FeatureTable.
    vsg::ref_ptr<Tiles3D::FeatureTable> createPropertyTable(gltf::SceneGraphBuilder& builder,
                                                            uint32_t index)
    {
        if (!builder.model) return {};

        auto ext = builder.model->extension<vsg::JSONtoMetaDataSchema>("EXT_structural_metadata");
        if (!ext || !ext->object) return {};

        // One table is routinely shared by many primitives -- there is a sample
        // named for it -- and decoding its columns again for each of them costs
        // the whole table per primitive.
        const std::string cacheKey = "vsgXchange.propertyTable." + std::to_string(index);
        if (auto cached = builder.model->getObject<Tiles3D::FeatureTable>(cacheKey))
            return vsg::ref_ptr<Tiles3D::FeatureTable>(cached);

        auto tables = metaArray(*ext->object, "propertyTables");
        if (!tables || index >= tables->children.size()) return {};

        auto table = tables->children[index];
        if (!table) return {};

        double count = 0.0;
        if (!metaNumber(*table, "count", count) || count <= 0.0) return {};

        // The class names the type of every column. Without it the bytes in the
        // bufferViews have no meaning at all.
        const std::string className = metaString(*table, "class");
        auto schema = metaObject(*ext->object, "schema");
        if (!schema) return {};
        auto classes = metaObject(*schema, "classes");
        if (!classes) return {};
        auto klass = className.empty() ? nullptr : metaObject(*classes, className.c_str());
        if (!klass) return {};
        auto classProperties = metaObject(*klass, "properties");
        if (!classProperties) return {};

        auto tableProperties = metaObject(*table, "properties");
        if (!tableProperties) return {};

        // Same caps as the 1.0 path, and for the same reason: every accepted
        // column is retained for the life of the tile and formatted on every
        // click, and all of it came from a document fetched over HTTP.
        constexpr size_t MAX_PROPERTIES = 128;
        constexpr size_t MAX_NAME_LENGTH = 256;

        auto features = Tiles3D::FeatureTable::create();
        features->count = uint32_t(count);

        // The properties a table declares are user objects on its metadata node,
        // so the map on the Auxiliary is the only way to enumerate them.
        auto aux = tableProperties->getAuxiliary();
        if (!aux) return {};

        for (auto& [name, value] : aux->userObjects)
        {
            if (features->properties.size() >= MAX_PROPERTIES)
            {
                vsg::warn("gltf: EXT_structural_metadata declares more than ", MAX_PROPERTIES,
                          " properties; the rest are dropped.");
                break;
            }
            if (name.size() > MAX_NAME_LENGTH) continue;

            auto tableProperty = value.cast<vsg::Object>();
            if (!tableProperty) continue;

            auto classProperty = dynamic_cast<vsg::Object*>(classProperties->getObject(name));
            if (!classProperty) continue;

            if (auto column = createPropertyColumn(builder, *classProperty, *tableProperty,
                                                   features->count, name))
            {
                // Shorter than the table says would be read past its end by
                // anything indexing it by feature.
                if (column->valueCount() < features->count)
                {
                    vsg::warn("gltf: EXT_structural_metadata property \"", name, "\" has ",
                              column->valueCount(), " values for ", features->count,
                              " features; it is dropped.");
                    continue;
                }
                features->properties[name] = column;
            }
        }

        if (features->properties.empty()) return {};

        builder.model->setObject(cacheKey, features);
        return features;
    }

    //! EXT_mesh_features: which attribute holds this primitive's ids, and which
    //! property table describes them.
    //!
    //! Only the first featureIds entry that names an ATTRIBUTE is used. A
    //! primitive may declare several, and one may be a feature-id TEXTURE --
    //! which needs a texture sample per pick and is a different mechanism
    //! entirely, so it is skipped rather than misread as an attribute index.
    bool readMeshFeatures(gltf::Primitive& primitive, int& out_attribute, int& out_propertyTable)
    {
        auto ext = primitive.extension<vsg::JSONtoMetaDataSchema>("EXT_mesh_features");
        if (!ext || !ext->object) return false;

        auto featureIds = metaArray(*ext->object, "featureIds");
        if (!featureIds) return false;

        for (auto& entry : featureIds->children)
        {
            if (!entry) continue;

            double attribute = 0.0;
            if (!metaNumber(*entry, "attribute", attribute)) continue;   // texture, or a constant
            if (attribute < 0.0) continue;

            out_attribute = int(attribute);

            double propertyTable = 0.0;
            out_propertyTable = metaNumber(*entry, "propertyTable", propertyTable) && propertyTable >= 0.0
                ? int(propertyTable)
                : -1;

            return true;
        }
        return false;
    }
}

vsg::ref_ptr<vsg::Node> gltf::SceneGraphBuilder::createMesh(vsg::ref_ptr<gltf::Mesh> gltf_mesh, const MeshExtras& meshExtras)
{
    /*
    struct Attributes : public vsg::Inherit<vsg::JSONParser::Schema, Attributes>
    {
        std::map<std::string, glTFid> values;
    };

    struct Primitive : public vsg::Inherit<ExtensionsExtras, Primitive>
    {
        Attributes attributes;
        glTFid indices;
        glTFid material;
        uint32_t mode = 0;
        vsg::ObjectsSchema<Attributes> targets;
    };

    struct Mesh : public vsg::Inherit<NameExtensionsExtras, Mesh>
    {
        vsg::ObjectsSchema<Primitive> primitives;
        vsg::ValuesSchema<double> weights;
    };
*/

    const VkPrimitiveTopology topologyLookup[] = {
        VK_PRIMITIVE_TOPOLOGY_POINT_LIST,     // 0, POINTS
        VK_PRIMITIVE_TOPOLOGY_LINE_LIST,      // 1, LINES
        VK_PRIMITIVE_TOPOLOGY_LINE_LIST,      // 2, LINE_LOOP, need special handling
        VK_PRIMITIVE_TOPOLOGY_LINE_STRIP,     // 3, LINE_STRIP
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,  // 4, TRIANGLES
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, // 5, TRIANGLE_STRIP
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN    // 6, TRIANGLE_FAN
    };
#if 0
    vsg::info("mesh = {");
    vsg::info("    primitives = ", gltf_mesh->primitives.values.size());
    vsg::info("    weight = ", gltf_mesh->weights.values.size());
#endif

    std::vector<vsg::ref_ptr<vsg::Node>> nodes;

    for (auto& primitive : gltf_mesh->primitives.values)
    {
        vsg::ref_ptr<vsg::DescriptorConfigurator> vsg_material;
        if (primitive->material)
        {
            vsg_material = vsg_materials[primitive->material.value];
        }
        else
        {
            vsg::debug("Material for primitive not assigned, primitive = ", primitive, ", primitive->material = ", primitive->material);
            vsg_material = default_material;
        }

        // A point cloud needs a vertex shader that writes gl_PointSize, and
        // that is the only thing the point shader set changes. Chosen by the
        // primitive's TOPOLOGY rather than by its material, so it covers both
        // routes: a .pnts, whose generated glTF is unlit, and a glTF that
        // simply declares mode 0 with an ordinary material.
        auto shaderSetForPrimitive = vsg_material->shaderSet;
        if (primitive->mode < (sizeof(topologyLookup) / sizeof(topologyLookup[0])) &&
            topologyLookup[primitive->mode] == VK_PRIMITIVE_TOPOLOGY_POINT_LIST)
        {
            if (auto points = getOrCreatePointShaderSet(vsg_material->shaderSet))
                shaderSetForPrimitive = points;
        }

        auto config = vsg::GraphicsPipelineConfigurator::create(shaderSetForPrimitive);
        config->descriptorConfigurator = vsg_material;
        if (options) config->assignInheritedState(options->inheritedState);

        if (meshExtras.jointSampler)
        {
            vsg_material->assignDescriptor("jointMatrices", meshExtras.jointSampler->jointMatrices);
        }

#if 0
        vsg::info("    primitive = {");
        vsg::info("        attributes = {");
        for(auto& [semantic, id] : primitive->attributes.values)
        {
             vsg::info("            ", semantic, ", ", id);
        }
        vsg::info("        }");
        vsg::info("        indices = ", primitive->indices);
        vsg::info("        material = ", primitive->material);
        vsg::info("        mode = ", primitive->mode);
        vsg::info("        targets = ", primitive->targets.values.size()) ;

        vsg::info("        * topology = ", topologyLookup[primitive->mode]) ;
        if (primitive->mode==2) vsg::info("        * LINE_LOOP needs special handling.");

        vsg::info("    }");
#endif

        vsg::DataList vertexArrays;

        // The shortest VERTEX-rate array seen, so the draw below cannot ask
        // for more vertices than the narrowest attribute actually has.
        uint32_t smallestVertexArray = std::numeric_limits<uint32_t>::max();

        auto assignArray = [&](Attributes& attrib, VkVertexInputRate vertexInputRate, const std::string& attribute_name) -> bool {
            auto array_itr = attrib.values.find(attribute_name);
            if (array_itr == attrib.values.end()) return false;

            auto name_itr = attributeLookup.find(attribute_name);
            if (name_itr == attributeLookup.end()) return false;

            if (array_itr->second.value >= vsg_accessors.size())
            {
                vsg::warn("gltf::SceneGraphBuilder::createMesh() error in assignArray( attrib, vertexIndexRate", attribute_name, "), array index out of range.");
                return false;
            }

            vsg::ref_ptr<vsg::Data> array = vsg_accessors[array_itr->second.value];
            if (!array)
            {
                vsg::warn("gltf::SceneGraphBuilder::createMesh() error in assignArray( attrib, vertexIndexRate", attribute_name, "), required array null.");
                return false;
            }

            if (attribute_name == "ROTATION")
            {
                if (auto vec4Rotations = array.cast<vsg::vec4Array>())
                {
                    // sizeof(vsg::quat), which is 16. The stride here was 12 --
                    // a vec3's -- so every instance rotation after the first
                    // read three floats from one quaternion and one from the
                    // next, and the last read four bytes past the end of the
                    // array. EXT_mesh_gpu_instancing's ROTATION is a VEC4
                    // quaternion; there is no 12-byte spelling of it.
                    //
                    // Found by sweeping the corpus for attributes whose format
                    // reads further than their stride advances, not by looking
                    // at this line: SimpleInstancing draws 125 cubes and the
                    // wrong ones are wrong by a rotation, which is not obvious
                    // in a still.
                    auto quatArray = vsg::quatArray::create(
                        array, 0, static_cast<uint32_t>(sizeof(vsg::quat)),
                        vec4Rotations->size());
                    array = quatArray;
                }
            }
            else if (attribute_name == "TEXCOORD_0" || attribute_name == "TEXCOORD_1" || attribute_name == "TEXCOORD_2" || attribute_name == "TEXCOORD_3")
            {
                // Widen first: a quantized texcoord arrives as usvec2 and the
                // transform below is written in floats, so doing it the other
                // way round silently skips the transform on quantized meshes.
                {
                    const bool normalized =
                        array_itr->second.value < model->accessors.values.size() &&
                        model->accessors.values[array_itr->second.value]->normalized;

                    if (auto widened = widenQuantizedAttribute(array, normalized))
                        array = widened;
                }

                if (auto texture_transform = vsg_material->getObject<KHR_texture_transform>("KHR_texture_transform"))
                {
                    vsg::vec2 offset(0.0f, 0.0f);
                    vsg::vec2 scale(1.0f, 1.0f);
                    float rotation = texture_transform->rotation;
                    if (texture_transform->offset.values.size() >= 2) offset.set(texture_transform->offset.values[0], texture_transform->offset.values[1]);
                    if (texture_transform->scale.values.size() >= 2) scale.set(texture_transform->scale.values[0], texture_transform->scale.values[1]);

                    if (auto texCoords = array.cast<vsg::vec2Array>())
                    {
                        float sin_rotation = std::sin(rotation);
                        float cos_rotation = std::cos(rotation);
                        auto transformedTexCoords = vsg::vec2Array::create(texCoords->size());
                        auto dest_itr = transformedTexCoords->begin();
                        for (auto& tc : *texCoords)
                        {
                            auto& dest_tc = *(dest_itr++);
                            dest_tc.x = offset.x + (tc.x * scale.x) * cos_rotation + (tc.y * scale.y) * sin_rotation;
                            dest_tc.y = offset.y + (tc.y * scale.y) * cos_rotation - (tc.x * scale.x) * sin_rotation;
                        }
                        array = transformedTexCoords;
                    }
                }
            }
            else if (attribute_name == "POSITION" || attribute_name == "NORMAL" ||
                     attribute_name == "TANGENT" ||
                     attribute_name == "COLOR_0" || attribute_name == "WEIGHTS_0")
            {
                // KHR_mesh_quantization. TEXCOORD_n is handled above, before
                // the texture transform, since that works in floats.
                //
                // COLOR_n and WEIGHTS_n are quantizable too and are consumed as
                // floats; JOINTS_n is NOT -- it is an index and stays integer,
                // which is why it has its own branch below rather than being
                // swept in here.
                const bool normalized =
                    array_itr->second.value < model->accessors.values.size() &&
                    model->accessors.values[array_itr->second.value]->normalized;

                if (auto widened = widenQuantizedAttribute(array, normalized))
                    array = widened;
            }
            else if (attribute_name == "JOINTS_0")
            {
                if (auto ushortCoords = array.cast<vsg::usvec4Array>())
                {
                    auto intCoords = vsg::ivec4Array::create(ushortCoords->size());
                    auto dest_itr = intCoords->begin();
                    for (auto& usc : *ushortCoords)
                    {
                        *(dest_itr++) = vsg::ivec4(usc[0], usc[1], usc[2], usc[3]);
                    }

                    array = intCoords;
                }
                else if (auto ubyteCoords = array.cast<vsg::ubvec4Array>())
                {
                    auto intCoords = vsg::ivec4Array::create(ubyteCoords->size());
                    auto dest_itr = intCoords->begin();
                    for (auto& ubc : *ubyteCoords)
                    {
                        *(dest_itr++) = vsg::ivec4(ubc[0], ubc[1], ubc[2], ubc[3]);
                    }

                    array = intCoords;
                }
            }

            setVertexFormatFromType(array);

            // The shortest VERTEX-rate array is what may safely be drawn.
            //
            // glTF requires every attribute accessor of a primitive to have the
            // same count, and the draw below takes its vertex count from the
            // FIRST array. A document that disagrees with itself -- POSITION
            // count 1000, NORMAL count 3 -- therefore had the pipeline read a
            // thousand normals out of a three-element array, which is a GPU
            // read far past the end of the buffer. The counts come from the
            // document, so this is the ordinary hostile case.
            if (vertexInputRate == VK_VERTEX_INPUT_RATE_VERTEX && array)
            {
                const uint32_t n = array->valueCount();
                if (n < smallestVertexArray) smallestVertexArray = n;
            }

            config->assignArray(vertexArrays, name_itr->second, vertexInputRate, array);
            return true;
        };

        if (!assignArray(primitive->attributes, VK_VERTEX_INPUT_RATE_VERTEX, "POSITION"))
        {
            vsg::warn("gltf::SceneGraphBuilder::createMesh() error no vertex array assigned.");
            return {};
        }

        if (!assignArray(primitive->attributes, VK_VERTEX_INPUT_RATE_VERTEX, "NORMAL"))
        {
            auto normal = vsg::vec3Value::create(vsg::vec3(0.0f, 0.0f, 1.0f));
            config->assignArray(vertexArrays, "vsg_Normal", VK_VERTEX_INPUT_RATE_INSTANCE, normal);
        }

        if (!assignArray(primitive->attributes, VK_VERTEX_INPUT_RATE_VERTEX, "TEXCOORD_0"))
        {
            auto texcoord = vsg::vec2Value::create(vsg::vec2(0.0f, 0.0f));
            config->assignArray(vertexArrays, "vsg_TexCoord0", VK_VERTEX_INPUT_RATE_INSTANCE, texcoord);
        }

        assignArray(primitive->attributes, VK_VERTEX_INPUT_RATE_VERTEX, "TEXCOORD_1");
        assignArray(primitive->attributes, VK_VERTEX_INPUT_RATE_VERTEX, "TEXCOORD_2");
        assignArray(primitive->attributes, VK_VERTEX_INPUT_RATE_VERTEX, "TEXCOORD_3");

        uint32_t vertexCount = vertexArrays.front()->valueCount();

        // Clamp rather than refuse. A count mismatch makes the document invalid
        // by the specification, but drawing the part that IS consistent is more
        // use than drawing nothing, and it is what the reader does everywhere
        // else it meets a document that disagrees with itself. What it must not
        // do is draw the part that is not there.
        if (smallestVertexArray != std::numeric_limits<uint32_t>::max() &&
            smallestVertexArray < vertexCount)
        {
            vsg::warn("gltf: a primitive's attribute accessors disagree on count -- "
                      "the first has ", vertexCount, " and the shortest has ",
                      smallestVertexArray, ". glTF requires them to be equal; drawing ",
                      smallestVertexArray, " vertices rather than reading past the shortest.");
            vertexCount = smallestVertexArray;
        }
        uint32_t instanceCount = 1;
        if (meshExtras.instancedAttributes)
        {
            for (auto& [name, id] : meshExtras.instancedAttributes->values)
            {
                instanceCount = vsg_accessors[id.value]->valueCount();
            }
        }

        if (!assignArray(primitive->attributes, VK_VERTEX_INPUT_RATE_VERTEX, "COLOR_0"))
        {
            if (instanceNodeHint == vsg::Options::INSTANCE_NONE)
            {
                auto defaultColor = vsg::vec4Array::create(instanceCount, vsg::vec4(1.0f, 1.0f, 1.0f, 1.0f));
                config->assignArray(vertexArrays, "vsg_Color", VK_VERTEX_INPUT_RATE_INSTANCE, defaultColor);
            }
            else if ((instanceNodeHint & vsg::Options::INSTANCE_COLORS) == 0)
            {
                auto defaultColor = vsg::vec4Array::create(vertexCount, vsg::vec4(1.0f, 1.0f, 1.0f, 1.0f));
                config->assignArray(vertexArrays, "vsg_Color", VK_VERTEX_INPUT_RATE_VERTEX, defaultColor);
            }
        }

        if (meshExtras.jointSampler)
        {
            assignArray(primitive->attributes, VK_VERTEX_INPUT_RATE_VERTEX, "JOINTS_0");
            assignArray(primitive->attributes, VK_VERTEX_INPUT_RATE_VERTEX, "WEIGHTS_0");
        }

        if (meshExtras.instancedAttributes)
        {
            assignArray(*meshExtras.instancedAttributes, VK_VERTEX_INPUT_RATE_INSTANCE, "TRANSLATION");
            assignArray(*meshExtras.instancedAttributes, VK_VERTEX_INPUT_RATE_INSTANCE, "ROTATION");
            assignArray(*meshExtras.instancedAttributes, VK_VERTEX_INPUT_RATE_INSTANCE, "SCALE");

            // The front face is PIPELINE state, so every instance in this draw
            // shares one winding. A per-instance SCALE whose components multiply
            // to a negative number mirrors that instance on its own, and no
            // single choice of front face can be right for both signs at once --
            // the ones on the wrong side get inverted gl_FrontFacing, inverted
            // culling, and a flipped normal under two-sided lighting.
            //
            // Fixing it needs the draw split into positive- and negative-handed
            // batches, which is a larger change than the node-level winding this
            // reader now implements. Say so rather than rendering it wrongly in
            // silence; 3D Tiles i3dm content in the wild is uniformly one sign.
            if (auto scale_itr = meshExtras.instancedAttributes->values.find("SCALE");
                scale_itr != meshExtras.instancedAttributes->values.end())
            {
                if (auto scales = vsg_accessors[scale_itr->second.value].cast<vsg::vec3Array>())
                {
                    bool anyPositive = false, anyNegative = false;
                    for (auto& s : *scales)
                    {
                        if (s.x * s.y * s.z < 0.0f) anyNegative = true;
                        else anyPositive = true;
                    }
                    if (anyNegative && anyPositive)
                    {
                        vsg::warn("EXT_mesh_gpu_instancing: instances of this mesh have "
                                  "both mirrored and unmirrored SCALE. They share one "
                                  "graphics pipeline, so one group will draw with the "
                                  "wrong winding; per-instance winding is not supported.");
                    }
                }
            }
        }

        vsg::ref_ptr<vsg::Node> draw;

        if (!meshExtras.instancedAttributes && instanceNodeHint != vsg::Options::INSTANCE_NONE)
        {
            if ((instanceNodeHint & vsg::Options::INSTANCE_COLORS) != 0) config->enableArray("vsg_Color", VK_VERTEX_INPUT_RATE_INSTANCE, 16, VK_FORMAT_R32G32B32A32_SFLOAT);
            if ((instanceNodeHint & vsg::Options::INSTANCE_TRANSLATIONS) != 0) config->enableArray("vsg_Translation", VK_VERTEX_INPUT_RATE_INSTANCE, 12, VK_FORMAT_R32G32B32_SFLOAT);
            if ((instanceNodeHint & vsg::Options::INSTANCE_ROTATIONS) != 0) config->enableArray("vsg_Rotation", VK_VERTEX_INPUT_RATE_INSTANCE, 16, VK_FORMAT_R32G32B32A32_SFLOAT);
            if ((instanceNodeHint & vsg::Options::INSTANCE_SCALES) != 0) config->enableArray("vsg_Scale", VK_VERTEX_INPUT_RATE_INSTANCE, 12, VK_FORMAT_R32G32B32_SFLOAT);

            if (primitive->indices)
            {
                auto instanceDrawIndexed = vsg::InstanceDrawIndexed::create();
                assign_extras(*primitive, *instanceDrawIndexed);
                instanceDrawIndexed->assignArrays(vertexArrays);

                auto indices = vsg_accessors[primitive->indices.value];
                if (!indices)
                {
                    vsg::warn("gltf::SceneGraphBuilder::createMesh() error required indices array null.");
                    return {};
                }

                {
                    uint32_t offender = 0;
                    if (!indicesAreInRange(indices, vertexCount, offender))
                    {
                        vsg::warn("gltf: a primitive's index ", offender, " is outside its ",
                                  vertexCount, " vertices; the primitive is skipped rather "
                                  "than drawn from memory past the end of the array.");
                        continue;
                    }
                }

                if (auto ubyte_indices = indices.cast<vsg::ubyteArray>())
                {
                    // need to promote ubyte indices to ushort as Vulkan requires an extension to be enabled for ubyte indices.
                    auto ushort_indices = vsg::ushortArray::create(ubyte_indices->size());
                    auto itr = ushort_indices->begin();
                    for (auto value : *ubyte_indices)
                    {
                        *(itr++) = static_cast<uint16_t>(value);
                    }

                    instanceDrawIndexed->assignIndices(ushort_indices);
                    instanceDrawIndexed->indexCount = static_cast<uint32_t>(ushort_indices->valueCount());
                }
                else
                {
                    instanceDrawIndexed->assignIndices(indices);
                    instanceDrawIndexed->indexCount = static_cast<uint32_t>(indices->valueCount());
                }
                draw = instanceDrawIndexed;
            }
            else
            {
                auto instanceDraw = vsg::InstanceDraw::create();
                assign_extras(*primitive, *instanceDraw);
                instanceDraw->assignArrays(vertexArrays);

                assign_extras(*primitive, *instanceDraw);
                instanceDraw->vertexCount = vertexCount;
                draw = instanceDraw;
            }
        }
        else if (primitive->indices)
        {
            auto vid = vsg::VertexIndexDraw::create();
            assign_extras(*primitive, *vid);
            vid->assignArrays(vertexArrays);
            vid->instanceCount = instanceCount;

            auto indices = vsg_accessors[primitive->indices.value];
            if (!indices)
            {
                vsg::warn("gltf::SceneGraphBuilder::createMesh() error required indices array null.");
                return {};
            }

            {
                uint32_t offender = 0;
                if (!indicesAreInRange(indices, vertexCount, offender))
                {
                    vsg::warn("gltf: a primitive's index ", offender, " is outside its ",
                              vertexCount, " vertices; the primitive is skipped rather "
                              "than drawn from memory past the end of the array.");
                    continue;
                }
            }

            if (auto ubyte_indices = indices.cast<vsg::ubyteArray>())
            {
                // need to promote ubyte indices to ushort as Vulkan requires an extension to be enabled for ubyte indices.
                auto ushort_indices = vsg::ushortArray::create(ubyte_indices->size());
                auto itr = ushort_indices->begin();
                for (auto value : *ubyte_indices)
                {
                    *(itr++) = static_cast<uint16_t>(value);
                }

                vid->assignIndices(ushort_indices);
                vid->indexCount = static_cast<uint32_t>(ushort_indices->valueCount());
            }
            else
            {
                vid->assignIndices(indices);
                vid->indexCount = static_cast<uint32_t>(indices->valueCount());
            }

            draw = vid;
        }
        else
        {
            auto vd = vsg::VertexDraw::create();
            assign_extras(*primitive, *vd);
            vd->assignArrays(vertexArrays);
            vd->instanceCount = instanceCount;
            vd->vertexCount = vertexCount;
            draw = vd;
        }

        // set the GraphicsPipelineStates to the required values.
        struct SetPipelineStates : public vsg::Visitor
        {
            VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            bool blending = false;
            bool two_sided = false;
            bool mirrored = false;

            SetPipelineStates(VkPrimitiveTopology in_topology, bool in_blending, bool in_two_sided, bool in_mirrored) :
                topology(in_topology), blending(in_blending), two_sided(in_two_sided), mirrored(in_mirrored) {}

            void apply(vsg::Object& object) { object.traverse(*this); }
            void apply(vsg::RasterizationState& rs)
            {
                if (two_sided) rs.cullMode = VK_CULL_MODE_NONE;

                // glTF 2.0 section 3.7.2.1: when the determinant of the node's
                // global transform is negative the winding order of the triangle
                // faces is reversed, so the front face becomes clockwise. This is
                // the remedy the reference implementations use -- CesiumJS swaps
                // renderState.cull.face on the determinant sign, and
                // cesium-unreal sets FMeshBatch::ReverseCulling from
                // IsLocalToWorldDeterminantNegative(). Neither rewrites the
                // geometry, which matters here: 3D Tiles content is routinely
                // Draco-compressed, and those indices cannot be reordered
                // without decoding and re-encoding the mesh.
                if (mirrored) rs.frontFace = VK_FRONT_FACE_CLOCKWISE;
            }
            void apply(vsg::InputAssemblyState& ias) { ias.topology = topology; }
            void apply(vsg::ColorBlendState& cbs) { cbs.configureAttachments(blending); }

        } sps(topologyLookup[primitive->mode], vsg_material->blending, vsg_material->two_sided, meshExtras.mirrored);

        config->accept(sps);

        if (sharedObjects)
            sharedObjects->share(config, [](auto gpc) { gpc->init(); });
        else
            config->init();

        // create StateGroup as the root of the scene/command graph to hold the GraphicsPipeline, and binding of Descriptors to decorate the whole graph
        auto stateGroup = vsg::StateGroup::create();

        config->copyTo(stateGroup, sharedObjects);

        stateGroup->addChild(draw);

        // Keep the per-vertex feature id.
        //
        // 3D Tiles marks which object each vertex belongs to with a custom
        // attribute -- `_BATCHID` in 1.0, `_FEATURE_ID_0` in 1.1 -- and neither
        // has a shader binding, so assignArray() drops both silently and a
        // picked triangle has no way back to the building it is part of.
        //
        // It is attached to the STATE GROUP rather than to the draw because a
        // ray intersection reports the path of nodes it passed through, and the
        // state group is on that path. The array is parallel to POSITION, so
        // the vertex index the intersection returns indexes it directly.
        // Which attribute, and which table describes it.
        //
        // EXT_mesh_features names both explicitly, and it is the only thing that
        // can: 1.1 allows several id sets on one primitive, and the one the
        // properties belong to is not necessarily _FEATURE_ID_0. Without the
        // extension -- every 1.0 tile, and the 1.1 documents that carry ids but
        // no table -- fall back to the conventional names.
        int featureAttribute = -1;
        int featurePropertyTable = -1;

        std::vector<std::string> semantics;
        if (readMeshFeatures(*primitive, featureAttribute, featurePropertyTable))
            semantics.push_back("_FEATURE_ID_" + std::to_string(featureAttribute));
        semantics.push_back("_BATCHID");
        semantics.push_back("_FEATURE_ID_0");

        for (const std::string& semantic : semantics)
        {
            auto itr = primitive->attributes.values.find(semantic);
            if (itr == primitive->attributes.values.end()) continue;
            if (!itr->second.valid() || itr->second.value >= vsg_accessors.size()) continue;

            auto ids = vsg_accessors[itr->second.value];
            if (!ids) continue;

            // Shorter than the vertices it labels means an index from a picked
            // triangle could fall outside it. The whole point of this array is
            // to be indexed by something that came from elsewhere.
            if (ids->valueCount() < vertexCount)
            {
                vsg::warn("gltf: ", semantic, " has ", ids->valueCount(),
                          " values for ", vertexCount, " vertices; it is dropped, so "
                          "this primitive's features cannot be identified.");
                continue;
            }

            stateGroup->setObject(Tiles3D::FEATURE_IDS_KEY, ids);

            // 1.1 keeps the properties in the DOCUMENT rather than in the tile
            // wrapper, so they are attached here, on the same node as the ids.
            // A 1.0 tile's properties arrive later, from the b3dm reader, and
            // land on this same node for the same reason: a consumer that finds
            // the two separately can pair a nested tile's ids with an
            // ancestor's table and report the wrong feature confidently.
            if (featurePropertyTable >= 0)
            {
                if (auto table = createPropertyTable(*this, uint32_t(featurePropertyTable)))
                    stateGroup->setObject(Tiles3D::FEATURE_TABLE_KEY, table);
            }

            break;      // 1.0 and 1.1 do not appear together; the first wins
        }

        if (vsg_material->blending)
        {
            if (meshExtras.instancedAttributes || instanceNodeHint != vsg::Options::INSTANCE_NONE)
            {
#if 0
                auto layer = vsg::Layer::create();
                layer->binNumber = 10;
                layer->child = stateGroup;

                nodes.push_back(layer);
#else
                nodes.push_back(stateGroup);
#endif
            }
            else
            {
                vsg::ComputeBounds computeBounds;
                draw->accept(computeBounds);
                vsg::dvec3 center = (computeBounds.bounds.min + computeBounds.bounds.max) * 0.5;
                double radius = vsg::length(computeBounds.bounds.max - computeBounds.bounds.min) * 0.5;

                auto depthSorted = vsg::DepthSorted::create();
                depthSorted->binNumber = 10;
                depthSorted->bound.set(center[0], center[1], center[2], radius);
                depthSorted->child = stateGroup;

                nodes.push_back(depthSorted);
            }
        }
        else
        {
            nodes.push_back(stateGroup);
        }

        // CESIUM_primitive_outline: draw the modeller's edges as lines.
        //
        // The extension names an accessor of vertex-index PAIRS -- the edges of
        // the model before it was triangulated. A cube's six quads become
        // twelve triangles with eighteen shared edges, and the extension says
        // which twelve of those a person actually drew.
        //
        // CesiumJS renders these by generating a per-vertex "outline
        // coordinate" and darkening fragments near an edge through a 1D texture
        // lookup, which needs a bespoke shader and can duplicate vertices. This
        // draws them as an actual LINE_LIST over the same positions instead:
        // less exact at glancing angles, a great deal less machinery, and the
        // edges are visibly there rather than absent. Said plainly here so the
        // difference from the reference is not mistaken for a bug.
        if (primitive->extensions)
        {
            auto outline_itr = primitive->extensions->values.find("CESIUM_primitive_outline");
            if (outline_itr != primitive->extensions->values.end() && outline_itr->second)
            {
                if (auto outline = outline_itr->second->cast<gltf::CESIUM_primitive_outline>())
                {
                    if (auto outlineNode = createPrimitiveOutline(primitive, *outline))
                    {
                        nodes.push_back(outlineNode);
                    }
                }
            }
        }
    }

    if (nodes.empty())
    {
        vsg::warn("Empty mesh");
        return {};
    }

    vsg::ref_ptr<vsg::Node> vsg_mesh;
    if (nodes.size() == 1)
    {
        // vsg::info("Mesh with single primtiive");
        vsg_mesh = nodes.front();
    }
    else
    {
        // vsg::info("Mesh with multiple primtiives - could possible use vsg::Geomterty.");
        auto group = vsg::Group::create();
        for (auto node : nodes)
        {
            group->addChild(node);
        }

        vsg_mesh = group;
    }

    assign_name_extras(*gltf_mesh, *vsg_mesh);

    return vsg_mesh;
}

bool gltf::SceneGraphBuilder::getTransform(gltf::Node& node, vsg::dmat4& matrix)
{
    if (node.matrix.values.size() == 16)
    {
        auto& m = node.matrix.values;
        matrix.set(m[0], m[1], m[2], m[3],
                   m[4], m[5], m[6], m[7],
                   m[8], m[9], m[10], m[11],
                   m[12], m[13], m[14], m[15]);
        return true;
    }
    else if (!(node.translation.values.empty()) ||
             !(node.rotation.values.empty()) ||
             !(node.scale.values.empty()))
    {
        auto& t = node.translation.values;
        auto& r = node.rotation.values;
        auto& s = node.scale.values;

        vsg::dvec3 vsg_t(0.0, 0.0, 0.0);
        vsg::dquat vsg_r;
        vsg::dvec3 vsg_s(1.0, 1.0, 1.0);

        if (t.size() >= 3) vsg_t.set(t[0], t[1], t[2]);
        if (r.size() >= 4) vsg_r.set(r[0], r[1], r[2], r[3]);
        if (s.size() >= 3) vsg_s.set(s[0], s[1], s[2]);

        matrix = vsg::translate(vsg_t) * vsg::rotate(vsg_r) * vsg::scale(vsg_s);
        return true;
    }
    else
    {
        return false;
    }
}

vsg::ref_ptr<vsg::Light> gltf::SceneGraphBuilder::createLight(vsg::ref_ptr<gltf::Light> gltf_light)
{
    bool range_set = gltf_light->range != std::numeric_limits<float>::max();

    vsg::ref_ptr<vsg::Light> vsg_light;
    if (gltf_light->type == "directional")
    {
        auto directionalLight = vsg::DirectionalLight::create();
        vsg_light = directionalLight;
    }
    else if (gltf_light->type == "point")
    {
        auto pointLight = vsg::PointLight::create();
        if (range_set) pointLight->radius = gltf_light->range;
        vsg_light = pointLight;
    }
    else if (gltf_light->type == "spot")
    {
        auto spotLight = vsg::SpotLight::create();
        if (range_set) spotLight->radius = gltf_light->range;
        vsg_light = spotLight;
        if (gltf_light->spot)
        {
            spotLight->innerAngle = gltf_light->spot->innerConeAngle;
            spotLight->outerAngle = gltf_light->spot->outerConeAngle;
        }
    }
    else
    {
        return {};
    }

    vsg_light->name = gltf_light->name;
    vsg_light->intensity = gltf_light->intensity;
    if (gltf_light->color.values.size() >= 3) vsg_light->color.set(gltf_light->color.values[0], gltf_light->color.values[1], gltf_light->color.values[2]);

    return vsg_light;
}

vsg::ref_ptr<vsg::Animation> gltf::SceneGraphBuilder::createAnimation(vsg::ref_ptr<gltf::Animation> gltf_animation)
{
    vsg::LogOutput log;

    auto vsg_animation = vsg::Animation::create();
    vsg_animation->name = gltf_animation->name;

    // gltf_animation->report(log);

    struct NodeChannels
    {
        vsg::ref_ptr<AnimationChannel> translation;
        vsg::ref_ptr<AnimationChannel> rotation;
        vsg::ref_ptr<AnimationChannel> scale;
        vsg::ref_ptr<AnimationChannel> weights;
    };

    std::map<uint32_t, NodeChannels> nodeChannels;

    for (auto& channel : gltf_animation->channels.values)
    {
        auto node_id = channel->target.node.value;

        if (channel->target.path == "translation")
            nodeChannels[node_id].translation = channel;
        else if (channel->target.path == "rotation")
            nodeChannels[node_id].rotation = channel;
        else if (channel->target.path == "scale")
            nodeChannels[node_id].scale = channel;
        else if (channel->target.path == "weights")
            nodeChannels[node_id].weights = channel;
        else
            vsg::warn("gltf::SceneGraphBuilder::createSceneGraph() unsupported AnimationChannel.target.path of ", channel->target.path);
    }

    for (auto& [node_id, channels] : nodeChannels)
    {
        if (channels.translation || channels.rotation || channels.scale)
        {
            auto keyframes = vsg::TransformKeyframes::create();

            if (channels.translation)
            {
                auto samplerID = channels.translation->sampler.value;
                auto sampler = gltf_animation->samplers.values[samplerID];
                auto vsg_input = vsg_accessors[sampler->input.value];
                auto vsg_output = vsg_accessors[sampler->output.value];

                auto timeValues = vsg_input.cast<vsg::floatArray>();
                auto translationValues = vsg_output.cast<vsg::vec3Array>();

                if (timeValues && translationValues)
                {
                    size_t count = std::min(vsg_input->valueCount(), vsg_output->valueCount());

                    auto& translations = keyframes->positions;
                    translations.resize(count);

                    for (size_t i = 0; i < count; ++i)
                    {
                        const auto& t = translationValues->at(i);
                        translations[i].time = timeValues->at(i);
                        translations[i].value.set(t.x, t.y, t.z);
                    }
                }
                else
                {
                    vsg::warn("gltf::SceneGraphBuilder::createAnimation(..) unsupported translation types. vsg_input = ", vsg_input, ", vsg_output = ", vsg_output);
                }
            }

            if (channels.rotation)
            {
                auto samplerID = channels.rotation->sampler.value;
                auto sampler = gltf_animation->samplers.values[samplerID];
                auto vsg_input = vsg_accessors[sampler->input.value];
                auto vsg_output = vsg_accessors[sampler->output.value];

                auto timeValues = vsg_input.cast<vsg::floatArray>();
                auto rotationValues = vsg_output.cast<vsg::vec4Array>();

                if (timeValues && rotationValues)
                {
                    size_t count = std::min(vsg_input->valueCount(), vsg_output->valueCount());

                    auto& rotations = keyframes->rotations;
                    rotations.resize(count);

                    for (size_t i = 0; i < count; ++i)
                    {
                        const auto& q = rotationValues->at(i);
                        rotations[i].time = timeValues->at(i);
                        rotations[i].value.set(q.x, q.y, q.z, q.w);
                    }
                }
                else
                {
                    vsg::warn("gltf::SceneGraphBuilder::createAnimation(..) unsupported rotation types. vsg_input = ", vsg_input, ", vsg_output = ", vsg_output);
                }
            }

            if (channels.scale)
            {
                auto samplerID = channels.scale->sampler.value;
                auto sampler = gltf_animation->samplers.values[samplerID];
                auto vsg_input = vsg_accessors[sampler->input.value];
                auto vsg_output = vsg_accessors[sampler->output.value];

                auto timeValues = vsg_input.cast<vsg::floatArray>();
                auto scaleValues = vsg_output.cast<vsg::vec3Array>();

                if (timeValues && scaleValues)
                {
                    size_t count = std::min(vsg_input->valueCount(), vsg_output->valueCount());

                    auto& scales = keyframes->scales;
                    scales.resize(count);

                    for (size_t i = 0; i < count; ++i)
                    {
                        const auto& s = scaleValues->at(i);
                        scales[i].time = timeValues->at(i);
                        scales[i].value.set(s.x, s.y, s.z);
                    }
                }
                else
                {
                    vsg::warn("gltf::SceneGraphBuilder::createAnimation(..) unsupported scale types. vsg_input = ", vsg_input, ", vsg_output = ", vsg_output);
                }
            }

            auto transformSampler = vsg::TransformSampler::create();

            auto node = vsg_nodes[node_id];
            if (auto mt = node.cast<vsg::MatrixTransform>())
            {
                vsg::decompose(mt->matrix, transformSampler->position, transformSampler->rotation, transformSampler->scale);
            }
            else if (auto joint = node.cast<vsg::Joint>())
            {
                vsg::decompose(joint->matrix, transformSampler->position, transformSampler->rotation, transformSampler->scale);
            }

            transformSampler->keyframes = keyframes;
            transformSampler->object = vsg_nodes[node_id];

            vsg_animation->samplers.push_back(transformSampler);
        }
    }

    return vsg_animation;
}

void gltf::SceneGraphBuilder::computeMirroredNodes()
{
    const size_t numNodes = model->nodes.values.size();
    node_mirrored.assign(numNodes, false);
    if (numNodes == 0) return;

    // Does a node's OWN transform mirror? A matrix does when the determinant of
    // its upper-left 3x3 is negative. A TRS does when the product of the scale
    // components is: a rotation is a unit quaternion and a translation is a
    // shear-free offset, so neither can change the sign.
    auto localMirrors = [](const gltf::Node& node) -> bool {
        const auto& m = node.matrix.values;
        if (m.size() >= 16)
        {
            // glTF matrices are column-major, so m[0..2] is the first column.
            double det = m[0] * (m[5] * m[10] - m[6] * m[9]) -
                         m[4] * (m[1] * m[10] - m[2] * m[9]) +
                         m[8] * (m[1] * m[6] - m[2] * m[5]);
            return det < 0.0;
        }

        const auto& s = node.scale.values;
        if (s.size() >= 3) return (s[0] * s[1] * s[2]) < 0.0;

        return false;
    };

    // Walk down from each root, iteratively. `visited` doubles as a cycle guard:
    // the specification forbids a cycle in `children`, but a reader that
    // recurses on trust hangs on a malformed file rather than reporting it.
    std::vector<bool> visited(numNodes, false);
    std::vector<std::pair<size_t, bool>> stack;

    auto walk = [&](size_t root) {
        stack.clear();
        stack.push_back({root, false});
        while (!stack.empty())
        {
            size_t ni = stack.back().first;
            bool parentMirrors = stack.back().second;
            stack.pop_back();

            if (ni >= numNodes || visited[ni]) continue;
            visited[ni] = true;

            auto& node = model->nodes.values[ni];
            if (!node) continue;

            // Exclusive-or and not "or": two mirrors cancel to a rotation.
            bool mirrors = (parentMirrors != localMirrors(*node));
            node_mirrored[ni] = mirrors;

            for (auto& id : node->children.values) stack.push_back({id.value, mirrors});
        }
    };

    for (auto& gltf_scene : model->scenes.values)
    {
        if (!gltf_scene) continue;
        for (auto& id : gltf_scene->nodes.values) walk(id.value);
    }

    // A node that no scene reaches is still built by the caller's loop, so give
    // it a defined answer rather than leaving the vector half filled.
    for (size_t ni = 0; ni < numNodes; ++ni)
    {
        if (!visited[ni]) walk(ni);
    }
}

vsg::ref_ptr<vsg::Node> gltf::SceneGraphBuilder::createNode(vsg::ref_ptr<gltf::Node> gltf_node, bool jointNode, bool mirrored)
{
    vsg::ref_ptr<vsg::Node> vsg_node;

    vsg::ref_ptr<vsg::Light> vsg_light;
    if (auto khr_lights = gltf_node->extension<KHR_lights_punctual>("KHR_lights_punctual"))
    {
        auto id = khr_lights->light;
        if (id && id.value < vsg_lights.size())
        {
            vsg_light = vsg_lights[id.value];
        }
    }

    MeshExtras meshExtras;

    if (gltf_node->skin)
    {
        meshExtras.jointSampler = vsg_skins[gltf_node->skin.value];
    }

    vsg::ref_ptr<vsg::Node> vsg_mesh;
    if (gltf_node->mesh)
    {
        // A mirrored node needs its own build of the mesh: the front face is
        // pipeline state, so the two windings cannot share one instance.
        meshExtras.mirrored = mirrored;
        auto& mesh_cache = mirrored ? vsg_meshes_mirrored : vsg_meshes;

        auto gltf_mesh = model->meshes.values[gltf_node->mesh.value];

        if (auto mesh_gpu_instancing = gltf_node->extension<EXT_mesh_gpu_instancing>("EXT_mesh_gpu_instancing"))
        {
            meshExtras.instancedAttributes = mesh_gpu_instancing->attributes;
        }

        if (!mesh_cache[gltf_node->mesh.value])
        {
            mesh_cache[gltf_node->mesh.value] = createMesh(gltf_mesh, meshExtras);
        }

        vsg_mesh = mesh_cache[gltf_node->mesh.value];
    }

    bool isTransform = !(gltf_node->matrix.values.empty()) ||
                       !(gltf_node->rotation.values.empty()) ||
                       !(gltf_node->scale.values.empty()) ||
                       !(gltf_node->translation.values.empty());

    size_t numChildren = gltf_node->children.values.size();
    if (gltf_node->camera) ++numChildren;
    if (vsg_mesh) ++numChildren;
    if (vsg_light) ++numChildren;

    if (jointNode)
    {
        auto joint = vsg::Joint::create();
        if (gltf_node->camera) joint->addChild(vsg_cameras[gltf_node->camera.value]);
        if (vsg_light) joint->addChild(vsg_light);
        if (vsg_mesh) joint->addChild(vsg_mesh);

        if (gltf_node->matrix.values.size() == 16)
        {
            auto& m = gltf_node->matrix.values;
            joint->matrix.set(m[0], m[1], m[2], m[3],
                              m[4], m[5], m[6], m[7],
                              m[8], m[9], m[10], m[11],
                              m[12], m[13], m[14], m[15]);
        }
        else
        {
            auto& t = gltf_node->translation.values;
            auto& r = gltf_node->rotation.values;
            auto& s = gltf_node->scale.values;

            vsg::dvec3 vsg_t(0.0, 0.0, 0.0);
            vsg::dquat vsg_r;
            vsg::dvec3 vsg_s(1.0, 1.0, 1.0);

            if (t.size() >= 3) vsg_t.set(t[0], t[1], t[2]);
            if (r.size() >= 4) vsg_r.set(r[0], r[1], r[2], r[3]);
            if (s.size() >= 3) vsg_s.set(s[0], s[1], s[2]);

            joint->matrix = vsg::translate(vsg_t) * vsg::rotate(vsg_r) * vsg::scale(vsg_s);
        }

        vsg_node = joint;
    }
    else if (isTransform)
    {
        auto transform = vsg::MatrixTransform::create();
        if (gltf_node->camera) transform->addChild(vsg_cameras[gltf_node->camera.value]);
        if (vsg_light) transform->addChild(vsg_light);
        if (vsg_mesh) transform->addChild(vsg_mesh);

        if (gltf_node->matrix.values.size() == 16)
        {
            auto& m = gltf_node->matrix.values;
            transform->matrix.set(m[0], m[1], m[2], m[3],
                                  m[4], m[5], m[6], m[7],
                                  m[8], m[9], m[10], m[11],
                                  m[12], m[13], m[14], m[15]);
        }
        else
        {
            auto& t = gltf_node->translation.values;
            auto& r = gltf_node->rotation.values;
            auto& s = gltf_node->scale.values;

            vsg::dvec3 vsg_t(0.0, 0.0, 0.0);
            vsg::dquat vsg_r;
            vsg::dvec3 vsg_s(1.0, 1.0, 1.0);

            if (t.size() >= 3) vsg_t.set(t[0], t[1], t[2]);
            if (r.size() >= 4) vsg_r.set(r[0], r[1], r[2], r[3]);
            if (s.size() >= 3) vsg_s.set(s[0], s[1], s[2]);

            transform->matrix = vsg::translate(vsg_t) * vsg::rotate(vsg_r) * vsg::scale(vsg_s);
        }

        vsg_node = transform;
    }
    else if (numChildren > 1 || gltf_node->requireMetaData())
    {
        auto group = vsg::Group::create();

        if (gltf_node->camera) group->addChild(vsg_cameras[gltf_node->camera.value]);
        if (vsg_light) group->addChild(vsg_light);
        if (vsg_mesh) group->addChild(vsg_mesh);

        vsg_node = group;
    }
    else
    {
        if (gltf_node->camera)
            vsg_node = vsg_cameras[gltf_node->camera.value];
        else if (vsg_mesh)
            vsg_node = vsg_mesh;
        else if (vsg_light)
            vsg_node = vsg_light;
        else
            vsg_node = vsg::Group::create(); // TODO: single child so should this just point to the child?
    }

    assign_name_extras(*gltf_node, *vsg_node);

    return vsg_node;
}

void gltf::SceneGraphBuilder::flattenTransforms(gltf::Node& node, const vsg::dmat4& inheritedTransform)
{
    vsg::dmat4 accumulatedTransform = inheritedTransform;

    vsg::dmat4 localMatrix;
    if (getTransform(node, localMatrix))
    {
        accumulatedTransform = accumulatedTransform * localMatrix;

        // clear node transform values so that they aren't reapplied.
        node.matrix.values.clear();
        node.rotation.values.clear();
        node.scale.values.clear();
        node.translation.values.clear();
    }

    if (node.camera)
    {
        vsg::info("TODO: need to flatten camera ", node.camera);
    }

    if (node.skin)
    {
        vsg::info("TODO: need to flatten skin ", node.skin);
    }

    auto inverse_accumulatedTransform = vsg::inverse(accumulatedTransform);

    if (node.mesh)
    {
        auto mesh = model->meshes.values[node.mesh.value];
        for (auto primitive : mesh->primitives.values)
        {
            if (auto position_itr = primitive->attributes.values.find("POSITION"); position_itr != primitive->attributes.values.end())
            {
                auto data = vsg_accessors[position_itr->second.value];
                auto vertices = data.cast<vsg::vec3Array>();

                for (auto& v : *vertices)
                {
                    v = accumulatedTransform * vsg::dvec3(v);
                }
            }
            if (auto normal_itr = primitive->attributes.values.find("NORMAL"); normal_itr != primitive->attributes.values.end())
            {
                auto data = vsg_accessors[normal_itr->second.value];
                auto normals = data.cast<vsg::vec3Array>();

                for (auto& n : *normals)
                {
                    n = vsg::dvec3(n) * inverse_accumulatedTransform;
                }
            }
        }
    }

    for (auto& id : node.children.values)
    {
        auto child = model->nodes.values[id.value];
        flattenTransforms(*child, accumulatedTransform);
    }
}

vsg::ref_ptr<vsg::Node> gltf::SceneGraphBuilder::createScene(vsg::ref_ptr<gltf::Scene> gltf_scene, bool requiresRootTransformNode, const vsg::dmat4& rootTransform)
{
    if (gltf_scene->nodes.values.empty())
    {
        vsg::warn("Cannot create scene graph from empty gltf::Scene.");
        return {};
    }

    vsg::Group::Children children;
    for (auto& id : gltf_scene->nodes.values)
    {
        if (vsg_nodes[id.value]) children.push_back(vsg_nodes[id.value]);
    }

    // add transform node if required
    if (requiresRootTransformNode)
    {
        auto transform = vsg::MatrixTransform::create(rootTransform);
        transform->children.swap(children);

        children.clear();
        children.push_back(transform);
    }

    // add animation group if required.
    if (!vsg_animations.empty())
    {
        auto animationGroup = vsg::AnimationGroup::create();
        animationGroup->animations = vsg_animations;

        animationGroup->children.swap(children);

        children.clear();
        children.push_back(animationGroup);
    }

    // All culling node if required.
    bool culling = vsg::value<bool>(true, gltf::culling, options) && (instanceNodeHint == vsg::Options::INSTANCE_NONE);
    if (culling)
    {
        if (auto bounds = vsg::visit<vsg::ComputeBounds>(children).bounds)
        {
            vsg::dsphere bs((bounds.max + bounds.min) * 0.5, vsg::length(bounds.max - bounds.min) * 0.5);
            if (children.size() == 1)
            {
                auto cullNode = vsg::CullNode::create(bs, children[0]);

                children.clear();
                children.push_back(cullNode);
            }
            else
            {
                auto cullGroup = vsg::CullGroup::create(bs);
                cullGroup->children.swap(children);

                children.clear();
                children.push_back(cullGroup);
            }
        }
    }

    if (children.size() > 1)
    {
        auto group = vsg::Group::create();
        group->children.swap(children);
        children.clear();
        children.push_back(group);
    }

    if (children.empty()) return {};

    vsg::ref_ptr<vsg::Node> vsg_scene = children[0];

    // assign meta data
    assign_name_extras(*gltf_scene, *vsg_scene);

    return vsg_scene;
}

#ifdef vsgXchange_draco
template<typename T>
static bool CopyDracoAttributes(const draco::PointAttribute* draco_attribute, void* ptr, draco::PointIndex::ValueType num_points)
{
    T* dest_ptr = reinterpret_cast<T*>(ptr);

    auto num_components = draco_attribute->num_components();
    for (draco::PointIndex i(0); i < num_points; ++i)
    {
        auto index = draco_attribute->mapped_index(i);
        if (!draco_attribute->ConvertValue(index, num_components, dest_ptr)) return false;

        dest_ptr += num_components;
    }

    return true;
}
#endif

bool gltf::SceneGraphBuilder::decodePrimitiveIfRequired(vsg::ref_ptr<gltf::Primitive> primitive)
{
    if (auto draco_mesh_compression = primitive->extension<KHR_draco_mesh_compression>("KHR_draco_mesh_compression"))
    {
#ifdef vsgXchange_draco
        auto& bufferView = model->bufferViews.values[draco_mesh_compression->bufferView.value];
        auto& buffer = model->buffers.values[bufferView->buffer.value];

        auto bufferViewData = static_cast<const char*>(buffer->data->dataPointer()) + bufferView->byteOffset;
        auto bufferViewSize = bufferView->byteLength;

        draco::DecoderBuffer decodeBuffer;
        decodeBuffer.Init(bufferViewData, bufferViewSize);

        draco::Decoder decoder;
        auto result = decoder.DecodeMeshFromBuffer(&decodeBuffer);

        auto& mesh = result.value();
        auto num_points = mesh->num_points();

        // process indices
        if (primitive->indices)
        {
            auto& indices = model->accessors.values[primitive->indices.value];

            // set the indices bufferView to the what will be the index value of bufferView to be created for the indices
            indices->bufferView.value = static_cast<uint32_t>(model->bufferViews.values.size());

            // hardwire to uint32_t for now
            uint32_t componentSize = (num_points < 65536) ? 2 : 4;
            uint32_t count = mesh->num_faces() * 3;

            // set up BufferView for the indices
            auto indexBufferView = gltf::BufferView::create();
            indexBufferView->buffer.value = static_cast<uint32_t>(model->buffers.values.size());
            indexBufferView->byteOffset = 0;
            indexBufferView->byteLength = count * componentSize;
            model->bufferViews.values.push_back(indexBufferView);

            // set up Buffer for the indices
            auto indexBuffer = gltf::Buffer::create();
            indexBuffer->byteLength = indexBufferView->byteLength;
            indexBuffer->data = vsg::ubyteArray::create(indexBuffer->byteLength);
            model->buffers.values.push_back(indexBuffer);

            if (componentSize == sizeof(draco::PointIndex))
            {
                indices->componentType = COMPONENT_TYPE_UNSIGNED_INT;
                indices->count = count;

                // compatible size so can just copy data directly
                memcpy(indexBuffer->data->dataPointer(), &(mesh->face(draco::FaceIndex(0)))[0], indexBuffer->byteLength);
            }
            else
            {
                indices->componentType = COMPONENT_TYPE_UNSIGNED_SHORT;
                indices->count = count;

                // copy data across value by value converting to ushort type
                uint16_t* dest = static_cast<uint16_t*>(indexBuffer->data->dataPointer());
                for (draco::FaceIndex i(0); i < mesh->num_faces(); ++i)
                {
                    const auto& face = mesh->face(i);
                    *(dest++) = static_cast<uint16_t>(face[0].value());
                    *(dest++) = static_cast<uint16_t>(face[1].value());
                    *(dest++) = static_cast<uint16_t>(face[2].value());
                }
            }
        }

        auto& draco_attributes = draco_mesh_compression->attributes.values;
        auto& primitive_attributes = primitive->attributes.values;

        for (auto& [name, id] : draco_attributes)
        {
            if (auto itr = primitive_attributes.find(name); itr != primitive_attributes.end())
            {
                auto& primitive_attribute = *itr;

                const auto draco_attribute = mesh->GetAttributeByUniqueId(id.value);
                auto& accessor = model->accessors.values[primitive_attribute.second.value];

                // update the attribute accessor to the decode entry
                accessor->bufferView.value = static_cast<uint32_t>(model->bufferViews.values.size());
                accessor->count = num_points;

                auto dataProperties = accessor->getDataProperties();
                uint32_t valueSize = dataProperties.componentSize * dataProperties.componentCount;

                // allocate buffer and bufferView for decoded attribute
                auto attributeBufferView = gltf::BufferView::create();
                attributeBufferView->buffer.value = static_cast<uint32_t>(model->buffers.values.size());
                attributeBufferView->byteLength = num_points * valueSize;
                attributeBufferView->byteOffset = draco_attribute->byte_offset();
                attributeBufferView->byteStride = draco_attribute->byte_stride();
                model->bufferViews.values.push_back(attributeBufferView);

                auto attributeBuffer = gltf::Buffer::create();
                attributeBuffer->byteLength = attributeBufferView->byteLength;
                attributeBuffer->data = vsg::ubyteArray::create(attributeBuffer->byteLength);
                model->buffers.values.push_back(attributeBuffer);

                auto* ptr = attributeBuffer->data->dataPointer();

                // TODO get the attributes from the draco mesh.
                switch (accessor->componentType)
                {
                case (COMPONENT_TYPE_BYTE):
                    CopyDracoAttributes<int8_t>(draco_attribute, ptr, num_points);
                    break;
                case (COMPONENT_TYPE_UNSIGNED_BYTE):
                    CopyDracoAttributes<uint8_t>(draco_attribute, ptr, num_points);
                    break;
                case (COMPONENT_TYPE_SHORT):
                    CopyDracoAttributes<int16_t>(draco_attribute, ptr, num_points);
                    break;
                case (COMPONENT_TYPE_UNSIGNED_SHORT):
                    CopyDracoAttributes<uint16_t>(draco_attribute, ptr, num_points);
                    break;
                case (COMPONENT_TYPE_INT):
                    CopyDracoAttributes<int32_t>(draco_attribute, ptr, num_points);
                    break;
                case (COMPONENT_TYPE_UNSIGNED_INT):
                    CopyDracoAttributes<uint32_t>(draco_attribute, ptr, num_points);
                    break;
                case (COMPONENT_TYPE_FLOAT):
                    CopyDracoAttributes<float>(draco_attribute, ptr, num_points);
                    break;
                case (COMPONENT_TYPE_DOUBLE):
                    CopyDracoAttributes<double>(draco_attribute, ptr, num_points);
                    break;
                default:
                    vsg::info("unsupported type ", dataProperties.componentType);
                    break;
                }
            }
        }

        return true;
#else
        vsg::info("Primitive draco_mesh_compression = ", draco_mesh_compression, " not supported.");
        return false;
#endif
    }
    return true;
}

vsg::ref_ptr<vsg::ShaderSet> gltf::SceneGraphBuilder::getOrCreatePbrShaderSet()
{
    if (pbrShaderSet) return pbrShaderSet;

    pbrShaderSet = vsg::createPhysicsBasedRenderingShaderSet(options);
    if (sharedObjects) sharedObjects->share(pbrShaderSet);

    return pbrShaderSet;
}

vsg::ref_ptr<vsg::Node> gltf::SceneGraphBuilder::createPrimitiveOutline(
    vsg::ref_ptr<gltf::Primitive> primitive, const gltf::CESIUM_primitive_outline& outline)
{
    auto position_itr = primitive->attributes.values.find("POSITION");
    if (position_itr == primitive->attributes.values.end()) return {};
    if (!position_itr->second.valid() || position_itr->second.value >= vsg_accessors.size()) return {};

    auto positionData = vsg_accessors[position_itr->second.value];
    if (!positionData) return {};

    // KHR_mesh_quantization stores POSITION as byte or short, and the widening
    // the mesh path does is to a LOCAL copy -- vsg_accessors still holds the
    // integer array. Casting straight to vec3Array therefore returns null for
    // every quantized model, and the outline would vanish with no message on
    // exactly the models most likely to have one, since quantization is what
    // mesh optimizers emit. Widen it here the same way.
    if (auto widened = widenQuantizedAttribute(
            positionData,
            position_itr->second.value < model->accessors.values.size() &&
                model->accessors.values[position_itr->second.value]->normalized))
    {
        positionData = widened;
    }

    auto positions = positionData.cast<vsg::vec3Array>();
    if (!positions || positions->empty())
    {
        vsg::warn("CESIUM_primitive_outline: POSITION is ",
                  positionData->className(), ", which the outline cannot use; "
                  "the outline is skipped.");
        return {};
    }

    if (!outline.indices.valid() || outline.indices.value >= vsg_accessors.size()) return {};
    auto rawIndices = vsg_accessors[outline.indices.value];
    if (!rawIndices) return {};

    // Vulkan indexes with 16- or 32-bit integers only, and the extension's
    // accessor may be any of the three unsigned widths. Widen a byte accessor
    // rather than declining it: an outline is decoration, and refusing to draw
    // one over an index width is a poor trade.
    vsg::ref_ptr<vsg::Data> indices;
    if (auto u32 = rawIndices.cast<vsg::uintArray>())
    {
        indices = u32;
    }
    else if (auto u16 = rawIndices.cast<vsg::ushortArray>())
    {
        indices = u16;
    }
    else if (auto u8 = rawIndices.cast<vsg::ubyteArray>())
    {
        auto widened = vsg::ushortArray::create(u8->size());
        auto dest = widened->begin();
        for (auto v : *u8) *(dest++) = static_cast<uint16_t>(v);
        indices = widened;
    }
    else if (auto s16 = rawIndices.cast<vsg::shortArray>())
    {
        // glTF says an index accessor is unsigned, but 3d-tiles-samples'
        // BoxPrimitiveOutline declares componentType 5122 -- SIGNED short --
        // and CesiumJS reads it anyway. Follow the reference and accept it,
        // while refusing a genuinely negative index rather than wrapping it
        // to 65535 and reading off the end of the vertex buffer.
        auto widened = vsg::ushortArray::create(s16->size());
        auto dest = widened->begin();
        for (auto v : *s16)
        {
            if (v < 0)
            {
                vsg::warn("CESIUM_primitive_outline: a signed index accessor holds ", v,
                          "; the outline is skipped.");
                return {};
            }
            *(dest++) = static_cast<uint16_t>(v);
        }
        indices = widened;
    }
    else if (auto s8 = rawIndices.cast<vsg::byteArray>())
    {
        auto widened = vsg::ushortArray::create(s8->size());
        auto dest = widened->begin();
        for (auto v : *s8)
        {
            if (v < 0)
            {
                vsg::warn("CESIUM_primitive_outline: a signed index accessor holds ", int(v),
                          "; the outline is skipped.");
                return {};
            }
            *(dest++) = static_cast<uint16_t>(v);
        }
        indices = widened;
    }
    else if (auto s32 = rawIndices.cast<vsg::intArray>())
    {
        auto widened = vsg::uintArray::create(s32->size());
        auto dest = widened->begin();
        for (auto v : *s32)
        {
            if (v < 0)
            {
                vsg::warn("CESIUM_primitive_outline: a signed index accessor holds ", v,
                          "; the outline is skipped.");
                return {};
            }
            *(dest++) = static_cast<uint32_t>(v);
        }
        indices = widened;
    }
    else
    {
        vsg::warn("CESIUM_primitive_outline: index accessor is ", rawIndices->className(),
                  ", which is not an integer array; the outline is skipped.");
        return {};
    }

    const uint32_t indexCount = static_cast<uint32_t>(indices->valueCount());
    if (indexCount < 2) return {};

    // An odd count means the last index has no partner, and a LINE_LIST draw
    // would read one past the end of the pairs. Drop the stray rather than
    // refusing the whole outline.
    const uint32_t lineIndexCount = indexCount - (indexCount % 2);

    // Every index must be inside the position array. These come from the
    // document, and Vulkan does not bounds-check an index buffer -- an
    // out-of-range index is a read of whatever follows the vertex buffer on
    // the GPU.
    bool inRange = true;
    const uint32_t vertexCount = static_cast<uint32_t>(positions->size());
    if (auto u32 = indices.cast<vsg::uintArray>())
    {
        for (uint32_t i = 0; i < lineIndexCount && inRange; ++i)
            if (u32->at(i) >= vertexCount) inRange = false;
    }
    else if (auto u16 = indices.cast<vsg::ushortArray>())
    {
        for (uint32_t i = 0; i < lineIndexCount && inRange; ++i)
            if (u16->at(i) >= vertexCount) inRange = false;
    }
    if (!inRange)
    {
        vsg::warn("CESIUM_primitive_outline: an edge index is outside the ",
                  vertexCount, " positions of its primitive; the outline is skipped.");
        return {};
    }

    auto shaderSet = getOrCreateFlatShaderSet();
    if (!shaderSet) return {};

    auto config = vsg::GraphicsPipelineConfigurator::create(shaderSet);
    if (options) config->assignInheritedState(options->inheritedState);

    // Position from the primitive; everything else a single constant at
    // INSTANCE rate, which is how the reader already supplies an attribute a
    // model does not carry.
    vsg::DataList vertexArrays;
    config->assignArray(vertexArrays, "vsg_Vertex", VK_VERTEX_INPUT_RATE_VERTEX, positions);
    config->assignArray(vertexArrays, "vsg_Normal", VK_VERTEX_INPUT_RATE_INSTANCE,
                        vsg::vec3Value::create(vsg::vec3(0.0f, 0.0f, 1.0f)));
    config->assignArray(vertexArrays, "vsg_TexCoord0", VK_VERTEX_INPUT_RATE_INSTANCE,
                        vsg::vec2Value::create(vsg::vec2(0.0f, 0.0f)));
    // Near-black rather than black: an edge should read as a drawn line, not as
    // a hole in the model.
    config->assignArray(vertexArrays, "vsg_Color", VK_VERTEX_INPUT_RATE_INSTANCE,
                        vsg::vec4Value::create(vsg::vec4(0.05f, 0.05f, 0.05f, 1.0f)));

    auto vid = vsg::VertexIndexDraw::create();
    vid->assignArrays(vertexArrays);
    vid->assignIndices(indices);
    vid->indexCount = lineIndexCount;
    vid->instanceCount = 1;

    struct SetOutlineStates : public vsg::Visitor
    {
        void apply(vsg::Object& object) override { object.traverse(*this); }
        void apply(vsg::InputAssemblyState& ias) override { ias.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST; }
        void apply(vsg::RasterizationState& rs) override
        {
            // The outline sits exactly on the surface it outlines, so half of
            // every edge loses the depth test to the triangles that share it
            // and the line comes out dashed. A small negative bias lifts it
            // toward the eye. Rasterized wide enough to survive at a distance
            // -- 1.0 is a single pixel and disappears the moment the model is
            // more than a few metres away.
            rs.cullMode = VK_CULL_MODE_NONE;
            rs.depthBiasEnable = VK_TRUE;
            rs.depthBiasConstantFactor = -1.0f;
            rs.depthBiasSlopeFactor = -1.0f;
            rs.lineWidth = 1.0f;
        }
    } sos;

    config->accept(sos);

    if (sharedObjects)
        sharedObjects->share(config, [](auto gpc) { gpc->init(); });
    else
        config->init();

    auto stateGroup = vsg::StateGroup::create();
    config->copyTo(stateGroup, sharedObjects);
    stateGroup->addChild(vid);

    return stateGroup;
}

vsg::ref_ptr<vsg::ShaderSet> gltf::SceneGraphBuilder::getOrCreatePointShaderSet(vsg::ref_ptr<vsg::ShaderSet> source)
{
    if (!source) source = vsg::createFlatShadedShaderSet(options);
    if (!source) return {};

    if (auto itr = pointShaderSets.find(source); itr != pointShaderSets.end())
        return itr->second;

    auto flat = source;

    // Clone before patching. Shader sets are shared -- vsg::SharedObjects hands
    // the same instance to every material of that kind in the scene -- so
    // editing one in place would give every TRIANGLE using it a point size too,
    // and change meshes that have nothing to do with this.
    auto patched = vsg::ShaderSet::create();
    patched->stages = flat->stages;
    patched->attributeBindings = flat->attributeBindings;
    patched->descriptorBindings = flat->descriptorBindings;
    patched->pushConstantRanges = flat->pushConstantRanges;
    patched->definesArrayStates = flat->definesArrayStates;
    patched->optionalDefines = flat->optionalDefines;
    patched->defaultGraphicsPipelineStates = flat->defaultGraphicsPipelineStates;
    patched->customDescriptorSetBindings = flat->customDescriptorSetBindings;

    // `variants` is deliberately NOT copied: it caches compiled stages against
    // the compile settings that produced them, and those were compiled from the
    // shader this one is a patch of.

    bool patchedAStage = false;

    for (auto& stage : patched->stages)
    {
        if (!stage || stage->stage != VK_SHADER_STAGE_VERTEX_BIT) continue;
        if (!stage->module || stage->module->source.empty()) continue;

        std::string source = stage->module->source;

        // Two edits, and both have to land or neither is any use: the
        // declaration in the out block and the assignment in main(). Both sit
        // behind #ifdef VSG_POINT_SPRITE, so rather than arrange for that
        // define to be set on every point material, the guards are removed --
        // this shader set is only ever used for POINT_LIST.
        const std::string declGuardOpen = "#ifdef VSG_POINT_SPRITE\n    float gl_PointSize;\n#endif";
        const std::string declPlain = "    float gl_PointSize;";

        const std::string assignGuarded =
            "#ifdef VSG_POINT_SPRITE\n    gl_PointSize = 1.0;\n#endif";

        // Size in pixels, not in metres.
        //
        // A world-sized point needs the viewport height to convert to pixels,
        // and this shader's push constants carry only the projection and
        // modelview -- so a "0.35 m" point would be fake precision computed
        // from an assumed 1080-pixel window. A fixed pixel size makes no claim
        // it cannot keep, is what most point-cloud viewers default to, and is
        // the difference between a cloud you can see and one you cannot.
        // Making it configurable, and world-sized, is VGIS-481's remainder.
        const std::string assignPlain = "    gl_PointSize = 4.0;";

        const bool hadDecl = source.find(declGuardOpen) != std::string::npos;
        const bool hadAssign = source.find(assignGuarded) != std::string::npos;

        if (!hadDecl || !hadAssign)
        {
            // The shader changed shape underneath us. Say so and use the flat
            // set unpatched: one-pixel points are a poor picture, a shader that
            // fails to compile is no picture at all.
            vsg::warn("gltf: could not find gl_PointSize in the flat vertex shader "
                      "(declaration ", hadDecl, ", assignment ", hadAssign,
                      "); point clouds will draw at one pixel per point.");
            continue;
        }

        source.replace(source.find(declGuardOpen), declGuardOpen.size(), declPlain);
        source.replace(source.find(assignGuarded), assignGuarded.size(), assignPlain);

        // A new module, with the SPIR-V dropped.
        //
        // A ShaderModule carries both the GLSL and its precompiled SPIR-V, and
        // the compiled code wins. Editing the source and keeping the code
        // produces a shader set that looks patched and behaves exactly as
        // before -- which is the kind of change that reads as "the fix did not
        // work" for an afternoon.
        auto module = vsg::ShaderModule::create(source);
        auto replacement = vsg::ShaderStage::create(stage->stage, stage->entryPointName, module);
        replacement->specializationConstants = stage->specializationConstants;
        stage = replacement;
        patchedAStage = true;
    }

    if (!patchedAStage)
    {
        // Nothing to gain and something to lose: hand back the original.
        pointShaderSets[source] = source;
        return source;
    }

    if (sharedObjects) sharedObjects->share(patched);

    pointShaderSets[source] = patched;
    return patched;
}

vsg::ref_ptr<vsg::ShaderSet> gltf::SceneGraphBuilder::getOrCreateFlatShaderSet()
{
    if (flatShaderSet) return flatShaderSet;

    flatShaderSet = vsg::createFlatShadedShaderSet(options);
    if (sharedObjects) sharedObjects->share(flatShaderSet);

    return flatShaderSet;
}

vsg::ref_ptr<vsg::Object> gltf::SceneGraphBuilder::createSceneGraph(vsg::ref_ptr<gltf::glTF> in_model, vsg::ref_ptr<const vsg::Options> in_options)
{
    model = in_model;
    if (!model) return {};

    // extensionsRequired means what it says: the document cannot be read
    // correctly without it. A codec is the case that matters -- the geometry is
    // not merely differently shaped, it is not there in a form anything here
    // can read, so the primitive is dropped and the model loads as an empty
    // scene that reports success. Two of the 292 models in glTF-Sample-Assets
    // were exactly that.
    //
    // Named here rather than silently, and only for what genuinely blocks the
    // read: KHR_mesh_quantization is required by six of those models too, and
    // it IS supported, so this list is deliberately short and specific.
    // glTF 2.0, 3.2: "If a loader does not support an extension listed in
    // extensionsRequired, it MUST fail to load the asset." Until now this loop
    // enforced that for exactly one extension, so a document requiring anything
    // else -- a geometry codec we do not have, or one that does not exist yet --
    // loaded and reported success with whatever the reader happened to salvage.
    //
    // The rule is applied here in two grades rather than one, and the split is
    // deliberate:
    //
    //   fullySupported   we implement it, so the document loads.
    //
    //   appearanceOnly   we do not implement it, but it changes only how a
    //                    surface LOOKS. The vertices, indices and transforms
    //                    are untouched, so the model is the right model, drawn
    //                    without a sheen term or a clearcoat layer. Refusing a
    //                    whole city model over that serves nobody, and the
    //                    warning says what was dropped.
    //
    //   anything else    refused. An unknown extension is unknown: it may be
    //                    the next mesh codec, in which case loading it means
    //                    drawing a model that is missing or wrong while
    //                    reporting success. That is the failure this whole
    //                    session has been about.
    //
    // Measured against the corpora -- and the first measurement was WRONG in a
    // way worth recording. It scanned standalone .gltf and .glb files and found
    // 14 distinct required extensions across 396 documents, all named below.
    // But 3D Tiles content is b3dm and i3dm: the glTF sits behind a header, and
    // that scan never decomposed one. Decoding the 101 embedded payloads in
    // 3d-tiles-samples turns up a fifteenth, KHR_techniques_webgl, required by
    // TilesetWithTreeBillboards -- which this rule duly refused, taking a
    // tileset that had worked all day with it.
    //
    // So: 15 distinct required extensions across both corpora, standalone and
    // embedded, and all 15 are named below.
    static const std::set<std::string> fullySupported = {
        "KHR_mesh_quantization",
        "KHR_texture_transform",
        "KHR_lights_punctual",
        "KHR_materials_unlit",
        "KHR_materials_specular",
        "KHR_materials_ior",
        "KHR_materials_pbrSpecularGlossiness",
        "KHR_materials_emissive_strength",
        "EXT_mesh_gpu_instancing",
        "CESIUM_primitive_outline",
#ifdef vsgXchange_draco
        "KHR_draco_mesh_compression",
#endif
#ifdef vsgXchange_meshoptimizer
        "EXT_meshopt_compression",
#endif
    };

    static const std::set<std::string> appearanceOnly = {
        // Material models we do not implement. Each one changes shading and
        // nothing else -- no attribute, no index, no transform.
        "KHR_materials_sheen",
        "KHR_materials_clearcoat",
        "KHR_materials_iridescence",
        "KHR_materials_anisotropy",
        "KHR_materials_transmission",
        "KHR_materials_volume",
        "KHR_materials_variants",
        "KHR_materials_dispersion",
        // glTF 1.0's shader-technique extension, still required by some 3D
        // Tiles 1.0 content -- 3d-tiles-samples' TilesetWithTreeBillboards is
        // one. It carries programs and techniques for MATERIALS; the
        // positions, indices and transforms underneath are untouched, so the
        // model draws with a default material instead of its authored shader.
        "KHR_techniques_webgl",
        // Texture container formats. An unreadable texture leaves the surface
        // untextured; the geometry underneath it is unaffected.
        "KHR_texture_basisu",
        "EXT_texture_webp",
        "EXT_texture_avif",
    };

    for (const auto& required : model->extensionsRequired.values)
    {
        if (required == "EXT_meshopt_compression")
        {
#ifdef vsgXchange_meshoptimizer
            // Decoded, but not all of it. The undecoded views are zero-filled,
            // so proceeding would hand the pipeline zeros where positions and
            // indices should be -- a model that draws, wrongly, and reports
            // success. When the document says it REQUIRES this extension, a
            // partial decode is a failed read.
            if (model->meshoptDecodeFailed)
            {
                vsg::warn("glTF requires ", required,
                          ", and at least one compressed bufferView could not be "
                          "decoded; the undecoded views hold zeros, so the model is "
                          "refused rather than drawn from them.");
                return {};
            }
#endif
        }

        if (fullySupported.count(required) != 0) continue;

        if (appearanceOnly.count(required) != 0)
        {
            vsg::warn("glTF requires ", required,
                      ", which this reader does not implement. It affects only how "
                      "the surface is shaded, so the model is loaded without it "
                      "rather than refused.");
            continue;
        }

        vsg::warn("glTF requires ", required,
                  ", which this reader does not implement and cannot assume is "
                  "harmless -- an unknown extension may be the one that holds the "
                  "geometry. The document is refused rather than drawn from "
                  "whatever could be salvaged.");
        return {};
    }

    if (in_options) options = in_options;

    if (options) sharedObjects = options->sharedObjects;
    if (!sharedObjects) sharedObjects = vsg::SharedObjects::create();

    instanceNodeHint = options ? options->instanceNodeHint : vsg::Options::INSTANCE_NONE;
    cloneAccessors = vsg::value<bool>(cloneAccessors, gltf::clone_accessors, options);
    maxAnisotropy = vsg::value<float>(maxAnisotropy, gltf::maxAnisotropy, options);

    // TODO: need to check that the glTF model is suitable for use of InstanceNode/InstanceDraw

    // vsg::info("gltf::SceneGraphBuilder::createSceneGraph() instanceNodeHint = ", instanceNodeHint);

    vsg::CoordinateConvention destination_coordinateConvention = vsg::CoordinateConvention::Z_UP;
    if (options) destination_coordinateConvention = options->sceneCoordinateConvention;

    vsg::dmat4 rootTransform;
    bool requiresRootTransformNode = vsg::transform(source_coordinateConvention, destination_coordinateConvention, rootTransform);

    if (!default_material)
    {
        default_material = vsg::DescriptorConfigurator::create();
        default_material->shaderSet = getOrCreatePbrShaderSet();

        auto pbrMaterialValue = vsg::PbrMaterialValue::create();
        auto& pbrMaterial = pbrMaterialValue->value();

        // defaults make surface grey and washed out, so reset them to provide something closer to glTF example viewer.
        pbrMaterial.metallicFactor = 0.0f;
        pbrMaterial.roughnessFactor = 0.0f;

        default_material->assignDescriptor("material", pbrMaterialValue);
    }

    for (size_t mi = 0; mi < model->meshes.values.size(); ++mi)
    {
        auto mesh = model->meshes.values[mi];
        for (auto primitive : mesh->primitives.values)
        {
            if (!decodePrimitiveIfRequired(primitive))
            {
                vsg::info("Reqires draco decompression but no support available.");
                return {};
            }
        }
    }

    vsg_buffers.resize(model->buffers.values.size());
    for (size_t bi = 0; bi < model->buffers.values.size(); ++bi)
    {
        vsg_buffers[bi] = createBuffer(model->buffers.values[bi]);
    }

    vsg_bufferViews.resize(model->bufferViews.values.size());
    for (size_t bvi = 0; bvi < model->bufferViews.values.size(); ++bvi)
    {
        vsg_bufferViews[bvi] = createBufferView(model->bufferViews.values[bvi]);
    }

    vsg_accessors.resize(model->accessors.values.size());
    for (size_t ai = 0; ai < model->accessors.values.size(); ++ai)
    {
        vsg_accessors[ai] = createAccessor(model->accessors.values[ai]);
    }

    // Which nodes mirror? glTF 2.0 (3.7.2.1) asks for the determinant of each
    // node's GLOBAL transform, but nodes are built further down in a flat loop
    // that has no parent on hand to ask, so accumulate the sign here, walking
    // down from the scene roots.
    //
    // THE POSITION OF THIS CALL IS THE WHOLE POINT. flattenTransforms(), just
    // below, bakes each node's transform into its vertex positions and then
    // CLEARS matrix/rotation/scale/translation. Run after it, this pre-pass sees
    // a tree with no transforms left in it and concludes that nothing mirrors --
    // and since i3dm.cpp sets instanceNodeHint for every instanced 3D Tiles
    // payload, that is precisely the content the rule is needed for. Baking a
    // reflection into the vertices reverses the winding just as surely as
    // applying it at draw time does, so the flag is right either way; it just
    // has to be read before the evidence is erased.
    computeMirroredNodes();

    if (instanceNodeHint != vsg::Options::INSTANCE_NONE)
    {
        requiresRootTransformNode = false;
        //vsg::info("Need to flatten all transforms");

        for (size_t sci = 0; sci < model->scenes.values.size(); ++sci)
        {
            auto gltf_scene = model->scenes.values[sci];
            for (auto& id : gltf_scene->nodes.values)
            {
                flattenTransforms(*(model->nodes.values[id.value]), rootTransform);
            }
        }
    }

    // vsg::info("create cameras = ", model->cameras.values.size());
    vsg_cameras.resize(model->cameras.values.size());
    for (size_t ci = 0; ci < model->cameras.values.size(); ++ci)
    {
        vsg_cameras[ci] = createCamera(model->cameras.values[ci]);
    }

    // set which nodes are joints
    vsg_joints.resize(model->nodes.values.size(), false);
    for (size_t si = 0; si < model->skins.values.size(); ++si)
    {
        auto& gltf_skin = model->skins.values[si];
        for (auto joint : gltf_skin->joints.values)
        {
            vsg_joints[joint.value] = true;
        }
    }

    // vsg::info("create samplers = ", model->samplers.values.size());
    vsg_samplers.resize(model->samplers.values.size());
    std::vector<uint32_t> maxDimensions(model->samplers.values.size(), 0);
    for (size_t sai = 0; sai < model->samplers.values.size(); ++sai)
    {
        vsg_samplers[sai] = createSampler(model->samplers.values[sai]);
    }

    // vsg::info("create images = ", model->images.values.size());
    vsg_images.resize(model->images.values.size());
    for (size_t ii = 0; ii < model->images.values.size(); ++ii)
    {
        if (model->images.values[ii]) vsg_images[ii] = createImage(model->images.values[ii]);
    }

    // vsg::info("create textures = ", model->textures.values.size());
    vsg_textures.resize(model->textures.values.size());
    for (size_t ti = 0; ti < model->textures.values.size(); ++ti)
    {
        auto& gltf_texture = model->textures.values[ti];
        auto& si = vsg_textures[ti] = createTexture(gltf_texture);

        if (si.sampler && si.image)
        {
            auto& maxDimension = maxDimensions[gltf_texture->sampler.value];
            if (si.image->width() > maxDimension) maxDimension = si.image->width();
            if (si.image->height() > maxDimension) maxDimension = si.image->height();
            if (si.image->depth() > maxDimension) maxDimension = si.image->depth();
        }
    }

    // reset the maxLod's to the be appropriate for the dimensions of the images being used.
    for (size_t sai = 0; sai < model->samplers.values.size(); ++sai)
    {
        float maxLod = std::floor(std::log2f(static_cast<float>(maxDimensions[sai])));
        if (vsg_samplers[sai]->maxLod > maxLod)
        {
            vsg_samplers[sai]->maxLod = maxLod;
        }
    }

    // vsg::info("create materials = ", model->materials.values.size());
    vsg_materials.resize(model->materials.values.size());
    for (size_t mi = 0; mi < model->materials.values.size(); ++mi)
    {
        vsg_materials[mi] = createMaterial(model->materials.values[mi]);
    }

    // vsg::info("create meshes = ", model->meshes.values.size());
    // populate vsg_meshes in the createNode method.
    vsg_meshes.resize(model->meshes.values.size());
    vsg_meshes_mirrored.resize(model->meshes.values.size());

    if (auto khr_lights = model->extension<KHR_lights_punctual>("KHR_lights_punctual"))
    {
        vsg_lights.resize(khr_lights->lights.values.size());
        for (size_t li = 0; li < khr_lights->lights.values.size(); ++li)
        {
            vsg_lights[li] = createLight(khr_lights->lights.values[li]);
        }
    }

    //vsg::info("create skins = ", model->skins.values.size(), ", model->nodes.values.size() = ", model->nodes.values.size());
    vsg_skins.resize(model->skins.values.size());
    for (size_t si = 0; si < model->skins.values.size(); ++si)
    {
        auto& gltf_skin = model->skins.values[si];

        auto jointSampler = vsg::JointSampler::create();
        jointSampler->jointMatrices = vsg::mat4Array::create(gltf_skin->joints.values.size());
        jointSampler->jointMatrices->properties.dataVariance = vsg::DYNAMIC_DATA;
        jointSampler->offsetMatrices.resize(gltf_skin->joints.values.size());

        auto& offsetMatrices = jointSampler->offsetMatrices;
        auto inverseBindMatrices = vsg_accessors[gltf_skin->inverseBindMatrices.value];
        if (auto floatMatrices = inverseBindMatrices.cast<vsg::mat4Array>())
        {
            for (size_t i = 0; i < gltf_skin->joints.values.size(); ++i)
            {
                offsetMatrices[i] = floatMatrices->at(i);
            }
        }
        else if (auto doubleMatrices = inverseBindMatrices.cast<vsg::dmat4Array>())
        {
            for (size_t i = 0; i < gltf_skin->joints.values.size(); ++i)
            {
                offsetMatrices[i] = doubleMatrices->at(i);
            }
        }

        //vsg::info("skin inverseBindMatrices = ", inverseBindMatrices, ", gltf_skin->joints.values.size() = ", gltf_skin->joints.values.size() );

        vsg_skins[si] = jointSampler;
        assign_name_extras(*gltf_skin, *jointSampler);
    }

    // vsg::info("create nodes = ", model->nodes.values.size());
    vsg_nodes.resize(model->nodes.values.size());
    for (size_t ni = 0; ni < model->nodes.values.size(); ++ni)
    {
        vsg_nodes[ni] = createNode(model->nodes.values[ni], vsg_joints[ni], node_mirrored[ni]);
    }

    for (size_t ni = 0; ni < model->nodes.values.size(); ++ni)
    {
        auto& gltf_node = model->nodes.values[ni];

        if (!gltf_node->children.values.empty())
        {

            if (auto vsg_group = vsg_nodes[ni].cast<vsg::Group>())
            {
                for (auto id : gltf_node->children.values)
                {
                    if (auto vsg_child = vsg_nodes[id.value])
                        vsg_group->addChild(vsg_child);
                    else
                        vsg::info("Unassigned vsg_child");
                }
            }
            else if (auto vsg_joint = vsg_nodes[ni].cast<vsg::Joint>())
            {
                for (auto id : gltf_node->children.values)
                {
                    if (auto vsg_child = vsg_nodes[id.value])
                        vsg_joint->addChild(vsg_child);
                    else
                        vsg::info("Unassigned vsg_child");
                }
            }
        }
    }

    //
    for (size_t si = 0; si < model->skins.values.size(); ++si)
    {
        auto& gltf_skin = model->skins.values[si];

        for (size_t i = 0; i < gltf_skin->joints.values.size(); ++i)
        {
            if (auto joint = vsg_nodes[gltf_skin->joints.values[i].value].cast<vsg::Joint>())
            {
                joint->index = i;
            }
        }

        if (gltf_skin->skeleton)
        {
            vsg_skins[si]->subgraph = vsg_nodes[gltf_skin->skeleton.value];
        }
        else if (!gltf_skin->joints.values.empty())
        {
            vsg_skins[si]->subgraph = vsg_nodes[gltf_skin->joints.values[0].value];
        }
    }

    // set up animations
    vsg_animations.resize(model->animations.values.size());
    for (size_t ai = 0; ai < model->animations.values.size(); ++ai)
    {
        vsg_animations[ai] = createAnimation(model->animations.values[ai]);

        // for now just add JointSampler to all animations, do need to check that animation is associted with samplers joints.
        for (auto& jointSampler : vsg_skins)
        {
            vsg_animations[ai]->samplers.push_back(jointSampler);
        }
    }

    // vsg::info("scene = ", model->scene);
    // vsg::info("scenes = ", model->scenes.values.size());

    vsg_scenes.resize(model->scenes.values.size());
    for (size_t sci = 0; sci < model->scenes.values.size(); ++sci)
    {
        vsg_scenes[sci] = createScene(model->scenes.values[sci], requiresRootTransformNode, rootTransform);
    }

    // create root node
    if (vsg_scenes.size() > 1)
    {
        auto vsg_switch = vsg::Switch::create();
        for (size_t sci = 0; sci < model->scenes.values.size(); ++sci)
        {
            auto& vsg_scene = vsg_scenes[sci];
            vsg_switch->addChild(true, vsg_scene);
        }

        vsg_switch->setSingleChildOn(model->scene.value);

        // vsg::info("Created a scenes with a switch");

        return vsg_switch;
    }
    else if (vsg_scenes.size() == 1)
    {
        // vsg::info("Created a single scene");
        return vsg_scenes.front();
    }
    else
    {
        vsg::info("Empty scene");
        return {};
    }
}
