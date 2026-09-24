import os

def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def safe_for_script_tag(s):
    # Prevent premature </script> closure if any embedded string contains it
    return s.replace("</script", "<\\/script").replace("</SCRIPT", "<\\/SCRIPT")

template = read("template.html")
data_json = safe_for_script_tag(read("data_blob.json"))
porter_js = safe_for_script_tag(read("porter_stemmer.browser.js"))
nlpcore_js = safe_for_script_tag(read("nlp_core.browser.js"))
icons_js = safe_for_script_tag(read("icons.js"))
app_js = safe_for_script_tag(read("app.js"))

out = template.replace("__DATA_JSON__", data_json)
out = out.replace("__PORTER_JS__", porter_js)
out = out.replace("__NLPCORE_JS__", nlpcore_js)
out = out.replace("__ICONS_JS__", icons_js)
out = out.replace("__APP_JS__", app_js)

assert "__DATA_JSON__" not in out
assert "__PORTER_JS__" not in out
assert "__NLPCORE_JS__" not in out
assert "__ICONS_JS__" not in out
assert "__APP_JS__" not in out

with open("index.html", "w", encoding="utf-8") as f:
    f.write(out)

with open("../index.html", "w", encoding="utf-8") as f:
    f.write(out)

print("Assembled index.html:", len(out), "bytes")
