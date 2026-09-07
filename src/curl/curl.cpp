/* <editor-fold desc="MIT License">

Copyright(c) 2021 Robert Osfield

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

</editor-fold> */

#include <vsg/io/read.h>
#include <vsgXchange/curl.h>

#include <curl/curl.h>

#include <fstream>
#include <iostream>
#include <algorithm>
#include <sstream>

using namespace vsgXchange;

namespace vsgXchange
{

    bool containsServerAddress(const vsg::Path& filename)
    {
        return filename.compare(0, 7, "http://") == 0 || filename.compare(0, 8, "https://") == 0;
    }

    /// Join a server directory to a relative sibling, with a FORWARD SLASH.
    ///
    /// vsg::Path::operator/ inserts the platform's native separator, and on
    /// Windows that is a backslash. Composing a URL with it produced
    ///
    ///     http://host:9000/.../LOD-0\Mesh.bin
    ///
    /// which is not a URL. curl could not fetch it, the read returned nothing,
    /// and the glTF reader went on to build a scene whose meshes had no vertex
    /// data -- so a tileset reported "payloads decoded, 0 failed" and drew
    /// nothing at all. The failure is silent at every level and only visible in
    /// the composed string, which nothing printed.
    ///
    /// This is Windows-only by nature: on Linux operator/ inserts '/' and the
    /// same code path works, which is why it went unnoticed.
    vsg::Path joinServerPath(const vsg::Path& serverDir, const vsg::Path& relative)
    {
        auto dir = serverDir.string();
        while (!dir.empty() && (dir.back() == '/' || dir.back() == '\\'))
            dir.pop_back();

        auto rel = relative.string();
        // A relative reference inside a glTF may itself use backslashes on a
        // document authored on Windows; a URL wants forward slashes throughout.
        std::replace(rel.begin(), rel.end(), '\\', '/');
        while (!rel.empty() && rel.front() == '/')
            rel.erase(rel.begin());

        return vsg::Path(dir + "/" + rel);
    }

    std::pair<vsg::Path, vsg::Path> getServerPathAndFilename(const vsg::Path& filename)
    {
        auto pos = filename.find("://");
        if (pos != vsg::Path::npos)
        {
            auto pos_slash = filename.find_first_of('/', pos + 3);
            if (pos_slash != vsg::Path::npos)
            {
                return {filename.substr(pos + 3, pos_slash - pos - 3), filename.substr(pos_slash + 1, vsg::Path::npos)};
            }
            else
            {
                return {filename.substr(pos + 3, vsg::Path::npos), ""};
            }
        }
        return {};
    }

    vsg::Path getFileCachePath(const vsg::Path& fileCache, const vsg::Path& filename)
    {
        auto pos = filename.find("://");
        if (pos != vsg::Path::npos)
        {
            return fileCache / filename.substr(pos + 3, vsg::Path::npos);
        }
        return {};
    }

    class curl::Implementation
    {
    public:
        Implementation();
        virtual ~Implementation();

        vsg::ref_ptr<vsg::Object> read(const vsg::Path& filename, vsg::ref_ptr<const vsg::Options> options = {}) const;

    protected:
    };

} // namespace vsgXchange

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// CURL ReaderWriter facade
//
curl::curl() :
    _implementation(nullptr)
{
}
curl::~curl()
{
    delete _implementation;
}
vsg::ref_ptr<vsg::Object> curl::read(const vsg::Path& filename, vsg::ref_ptr<const vsg::Options> options) const
{
    vsg::Path serverFilename = filename;

    bool contains_serverAddress = containsServerAddress(filename);


    if (options)
    {
        if (!contains_serverAddress && !options->paths.empty())
        {
            contains_serverAddress = containsServerAddress(options->paths.front());
            if (contains_serverAddress)
            {
                serverFilename = joinServerPath(options->paths.front(), filename);
            }
        }

        if (contains_serverAddress && options->fileCache)
        {
            auto fileCachePath = getFileCachePath(options->fileCache, serverFilename);
            if (vsg::fileExists(fileCachePath))
            {
                auto local_options = vsg::clone(options);

                local_options->paths.insert(local_options->paths.begin(), vsg::filePath(serverFilename));
                local_options->extensionHint = vsg::lowerCaseFileExtension(filename);

                std::ifstream fin(fileCachePath, std::ios::in | std::ios::binary);
                auto object = vsg::read(fin, local_options); // do we need to remove any http URL?
                if (object)
                {
                    return object;
                }
            }
        }
    }

    if (contains_serverAddress)
    {
        {
            std::scoped_lock<std::mutex> lock(_mutex);
            if (!_implementation) _implementation = new curl::Implementation();
        }

        return _implementation->read(serverFilename, options);
    }
    else
    {
        return {};
    }
}

bool curl::getFeatures(Features& features) const
{
    features.protocolFeatureMap["http"] = vsg::ReaderWriter::READ_FILENAME;
    features.protocolFeatureMap["https"] = vsg::ReaderWriter::READ_FILENAME;
    features.optionNameTypeMap[curl::SSL_OPTIONS] = "uint32_t";
    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// CURL ReaderWriter implementation
//

// use static mutex and counter to track whether the curl_global_init(..) and curl_global_cleanup() should be called.
bool curl::s_do_curl_global_init_and_cleanup = true;
std::mutex s_curlImplementationMutex;
uint32_t s_curlImplementationCount = 0;

curl::Implementation::Implementation()
{
    if (curl::s_do_curl_global_init_and_cleanup)
    {
        std::scoped_lock<std::mutex> lock(s_curlImplementationMutex);
        if (s_curlImplementationCount == 0)
        {
            //std::cout<<"curl_global_init()"<<std::endl;
            curl_global_init(CURL_GLOBAL_ALL);
        }
        ++s_curlImplementationCount;
    }
}

curl::Implementation::~Implementation()
{
    if (s_do_curl_global_init_and_cleanup)
    {
        std::scoped_lock<std::mutex> lock(s_curlImplementationMutex);
        --s_curlImplementationCount;
        if (s_curlImplementationCount == 0)
        {
            //std::cout<<"curl_global_cleanup()"<<std::endl;
            curl_global_cleanup();
        }
    }
}

size_t StreamCallback(void* ptr, size_t size, size_t nmemb, void* user_data)
{
    size_t realsize = size * nmemb;

    if (user_data)
    {
        std::ostream* ostr = reinterpret_cast<std::ostream*>(user_data);
        ostr->write(reinterpret_cast<const char*>(ptr), realsize);
    }

    return realsize;
}

vsg::ref_ptr<vsg::Object> curl::Implementation::read(const vsg::Path& filename, vsg::ref_ptr<const vsg::Options> options) const
{
    auto _curl = curl_easy_init();

    curl_easy_setopt(_curl, CURLOPT_USERAGENT, "libcurl-agent/1.0"); // make user controllable?
    curl_easy_setopt(_curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(_curl, CURLOPT_WRITEFUNCTION, StreamCallback);

    std::stringstream sstr;

    curl_easy_setopt(_curl, CURLOPT_URL, filename.string().c_str());
    curl_easy_setopt(_curl, CURLOPT_WRITEDATA, (void*)&sstr);

#if CURL_AT_LEAST_VERSION(7, 21, 6)
    // "" means any encoding curl supports
    curl_easy_setopt(_curl, CURLOPT_ACCEPT_ENCODING, "");
#else
    curl_easy_setopt(_curl, CURLOPT_ENCODING, "");
#endif

    uint32_t sslOptions = 0;
#if defined(_WIN32) && defined(CURLSSLOPT_NATIVE_CA)
    sslOptions |= CURLSSLOPT_NATIVE_CA;
#endif
    if (options) options->getValue(curl::SSL_OPTIONS, sslOptions);
    if (sslOptions != 0) curl_easy_setopt(_curl, CURLOPT_SSL_OPTIONS, sslOptions);

    vsg::ref_ptr<vsg::Object> object;

    CURLcode result = curl_easy_perform(_curl);
    if (result == 0)
    {
        // https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
        long response_code = 0;
        result = curl_easy_getinfo(_curl, CURLINFO_RESPONSE_CODE, &response_code);

        if (result == 0 && response_code >= 200 && response_code < 300) // successful responses.
        {
            // success
            auto local_options = vsg::clone(options);
            local_options->paths.insert(local_options->paths.begin(), vsg::filePath(filename));

            // The URL's own extension wins over an inherited hint.
            //
            // extensionHint exists to type a stream that has no filename. Here
            // there IS one, and it describes THIS resource; a hint reaching us
            // from a caller describes the document that referenced it. Keeping
            // the hint meant a .gltf fetched over http had its sibling .bin
            // buffers fetched successfully and then handed to the glTF reader,
            // which reported "Unable to open file" for a request that returned
            // 200. The symptom was a tile that drew nothing, with eight "no
            // vsg::Data available to create BufferView" lines and no failed
            // request anywhere to point at.
            //
            // A hint still applies when the URL says nothing -- a
            // template-expanded content address with no extension, which is
            // ordinary in 3D Tiles.
            if (auto ext = vsg::lowerCaseFileExtension(filename); ext)
            {
                local_options->extensionHint = ext;
            }
            else if (!local_options->extensionHint)
            {
                local_options->extensionHint = ext;
            }

            object = vsg::read(sstr, local_options);

            if (object && options->fileCache)
            {
                auto fileCachePath = getFileCachePath(options->fileCache, filename);
                if (fileCachePath)
                {
                    vsg::makeDirectory(vsg::filePath(fileCachePath));

                    // reset the stringstream iterator to the beginning so we can copy it to the file cache file.
                    sstr.clear(std::stringstream::goodbit);
                    sstr.seekg(0);

                    std::ofstream fout(fileCachePath, std::ios::out | std::ios::binary);

                    fout << sstr.rdbuf();
                }
            }
        }
        else
        {
            object = vsg::ReadError::create(vsg::make_string("vsgXchange::curl could not read file ", filename, ", CURLINFO_RESPONSE_CODE = ", response_code));
        }
    }
    else
    {
        object = vsg::ReadError::create(vsg::make_string("vsgXchange::curl could not read file ", filename, ", result = ", result, ", ", curl_easy_strerror(result)));
    }

    if (_curl) curl_easy_cleanup(_curl);

    return object;
}
