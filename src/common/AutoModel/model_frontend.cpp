/// \file model_frontend.cpp
/// \brief Frontend registry
#include "AutoModel/model_frontend.hpp"

#include <sstream>
#include <stdexcept>
#include <utility>

namespace flm::frontend {
namespace {

std::string Join(const std::vector<std::string>& families) {
    if (families.empty()) return "(none)";
    std::ostringstream out;
    for (std::size_t i = 0; i < families.size(); ++i) {
        if (i) out << ", ";
        out << families[i];
    }
    return out.str();
}

void DestroyFrontend(void (*destroy)(void*, void*),
                     void* context, void* object) noexcept {
    if (!destroy || !object) return;
    try {
        destroy(context, object);
    } catch (...) {
        // Destruction has no error channel.
    }
}

}  // namespace

void FrontendDeleter::operator()(AutoModel* frontend) const noexcept {
    DestroyFrontend(destroy, context, static_cast<void*>(frontend));
}

FrontendRegistry::~FrontendRegistry() = default;

FrontendRegistry& FrontendRegistry::instance() {
    static FrontendRegistry registry;
    static std::once_flag once;
    // register_builtin_frontends must not call instance(), or this deadlocks.
    std::call_once(once, [] { register_builtin_frontends(registry); });
    return registry;
}

void FrontendRegistry::register_frontend(std::string family,
                                         FrontendFactory factory) {
    if (family.empty()) throw std::runtime_error("frontend family is empty");
    if (!factory) throw std::runtime_error("frontend factory is null");

    std::lock_guard lock(mutex_);
    if (factories_.count(family)) {
        throw std::runtime_error(
            "frontend family '" + family + "' is already registered");
    }
    factories_.emplace(std::move(family), std::move(factory));
}

bool FrontendRegistry::has(const std::string& family) const {
    std::lock_guard lock(mutex_);
    return factories_.count(family) != 0;
}

std::vector<std::string> FrontendRegistry::available() const {
    std::lock_guard lock(mutex_);
    std::vector<std::string> families;
    families.reserve(factories_.size());
    for (const auto& [family, factory] : factories_) {
        (void)factory;
        families.push_back(family);
    }
    return families;
}

FrontendPtr FrontendRegistry::create(const std::string& family,
                                     void* device) const {
    FrontendFactory factory;
    {
        std::lock_guard lock(mutex_);
        const auto found = factories_.find(family);
        if (found == factories_.end()) {
            std::vector<std::string> families;
            families.reserve(factories_.size());
            for (const auto& [registered, entry] : factories_) {
                (void)entry;
                families.push_back(registered);
            }
            throw std::runtime_error(
                "Model family '" + family +
                "' has no frontend. This build provides: " +
                Join(families));
        }
        factory = found->second;
    }
    auto frontend = factory(device);
    if (!frontend) {
        throw std::runtime_error(
            "Frontend factory for model family '" + family +
            "' returned null");
    }
    return frontend;
}

}  // namespace flm::frontend
