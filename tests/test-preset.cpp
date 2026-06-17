#include "preset.h"
#include "arg.h"
#include "testing.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

static void test_get_model_id(testing & t) {
    common_preset_context ctx(LLAMA_EXAMPLE_SERVER);

    t.test("get_model_id identity", [&](testing & t) {
        common_preset p1;
        p1.set_option(ctx, "LLAMA_ARG_MODEL", "model_a");
        p1.set_option(ctx, "LLAMA_ARG_MODEL_URL", "http://example.com");

        common_preset p2;
        p2.set_option(ctx, "LLAMA_ARG_MODEL", "model_a");
        p2.set_option(ctx, "LLAMA_ARG_MODEL_URL", "http://example.com");

        t.assert_equal(p1.get_model_id(), p2.get_model_id());
    });

    t.test("get_model_id collision resistance", [&](testing & t) {
        // test collision resistance with special characters
        common_preset p1;
        p1.set_option(ctx, "LLAMA_ARG_MODEL", "path;LLAMA_ARG_MODEL_URL=evil");
        p1.set_option(ctx, "LLAMA_ARG_MODEL_URL", "none");

        common_preset p2;
        p2.set_option(ctx, "LLAMA_ARG_MODEL", "path");
        p2.set_option(ctx, "LLAMA_ARG_MODEL_URL", "evil;LLAMA_ARG_MODEL_URL=none");

        t.assert_true(p1.get_model_id() != p2.get_model_id());
    });

    t.test("get_model_id distinct models", [&](testing & t) {
        common_preset p1;
        p1.set_option(ctx, "LLAMA_ARG_MODEL", "model_a");

        common_preset p2;
        p2.set_option(ctx, "LLAMA_ARG_MODEL", "model_b");

        t.assert_true(p1.get_model_id() != p2.get_model_id());
    });
}

static void test_to_json_sampling(testing & t) {
    common_preset_context ctx(LLAMA_EXAMPLE_SERVER);

    t.test("to_json_sampling basic", [&](testing & t) {
        common_preset p;
        p.set_option(ctx, "temp", "0.7");
        p.set_option(ctx, "LLAMA_ARG_TOP_K", "40");
        p.set_option(ctx, "LLAMA_ARG_REASONING", "true");

        std::string json_str = p.to_json_sampling();
        json j = json::parse(json_str);

        t.assert_equal(0.7, j["temperature"].get<double>());
        t.assert_equal(40, j["top_k"].get<int>());
        t.assert_equal(true, j["chat_template_kwargs"]["enable_thinking"].get<bool>());
    });

    t.test("to_json_sampling reasoning false", [&](testing & t) {
        common_preset p;
        p.set_option(ctx, "LLAMA_ARG_REASONING", "false");

        std::string json_str = p.to_json_sampling();
        json j = json::parse(json_str);

        t.assert_equal(false, j["chat_template_kwargs"]["enable_thinking"].get<bool>());
    });

    t.test("to_json_sampling reasoning auto", [&](testing & t) {
        common_preset p;
        p.set_option(ctx, "LLAMA_ARG_REASONING", "auto");

        std::string json_str = p.to_json_sampling();
        json j = json::parse(json_str);

        t.assert_true(!j.contains("chat_template_kwargs"));
    });
}

int main() {
    testing t;

    test_get_model_id(t);
    test_to_json_sampling(t);

    return t.summary();
}
