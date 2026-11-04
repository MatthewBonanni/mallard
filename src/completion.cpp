/**
 * @file completion.cpp
 * @brief `mallard completion bash|zsh|fish`: shell completion scripts made from the command table.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "build_info.h"
#include "cli.h"

namespace cli {

namespace {

/**
 * What completes an option's value or a positional argument, from its placeholder
 * ("FILE|latest", "[DIRECTORY]"): lowercase alternatives are literal words; files,
 * directories and templates are kinds, written "@files[:GLOB]", "@dirs", "@templates".
 */
using Values = std::vector<std::string>;

struct Flag {
    std::vector<std::string> names;  ///< name, then alias
    bool takes_value = false;        ///< as the next argument (or attached with =)
    bool attached = false;           ///< only as --name=VALUE
    Values values;
    std::string help;
};

struct Node {
    std::string path;  ///< "", "input", "input upgrade"
    std::vector<std::pair<std::string, std::string>> children;  ///< next word, summary
    std::vector<Flag> options;
    std::vector<Values> positional;
    bool variadic = false;
};

/** @brief File types an option's file always is, whatever the command. */
constexpr std::pair<std::string_view, std::string_view> FILE_GLOBS[] = {{"-i", "*.toml"}, {"--restart", "*.restart"}};

bool literal(std::string_view word) {
    return !word.empty() && std::all_of(word.begin(), word.end(), [](char c) {
        return std::islower(static_cast<unsigned char>(c)) || std::isdigit(static_cast<unsigned char>(c)) || c == '-' ||
               c == '_';
    });
}

std::vector<std::string> split(std::string_view text, char separator) {
    std::vector<std::string> out;
    size_t start = 0;
    for (size_t i = 0; i <= text.size(); i++) {
        if (i == text.size() || text[i] == separator) {
            out.emplace_back(text.substr(start, i - start));
            start = i + 1;
        }
    }
    return out;
}

Values values_of(std::string_view spec, std::string_view option = {}) {
    std::string s(spec);
    if (s.starts_with('[')) s = s.substr(1);
    if (s.ends_with("...]")) s.resize(s.size() - 4);
    if (s.ends_with(']')) s.pop_back();
    if (s.starts_with('=')) s = s.substr(1);
    Values v;
    for (const std::string & alt : split(s, '|')) {
        if (literal(alt)) {
            v.push_back(alt);
        } else if (alt.ends_with("DIR") || alt.ends_with("DIRECTORY")) {
            v.push_back("@dirs");
        } else if (alt == "TEMPLATE") {
            v.push_back("@templates");
        } else if (alt == "FILE" || alt == "INPUT") {
            std::string token = "@files";
            for (const auto & [name, glob] : FILE_GLOBS) {
                if (name == option) token += ":" + std::string(glob);
            }
            v.push_back(token);
        } else if (const size_t dot = alt.rfind('.');
                   dot != std::string::npos && alt.find_first_of(":=") == std::string::npos) {
            v.push_back("@files:*" + alt.substr(dot));
        }
    }
    return v;
}

/** @brief First sentence of a help text. */
std::string brief(std::string_view help) {
    const size_t end = help.find(". ");
    std::string out(help.substr(0, end));
    if (out.ends_with('.')) out.pop_back();
    return out;
}

void add_option(std::vector<Flag> & out, const Option & o) {
    if (o.name.ends_with('*')) {
        // A prefix row: the options its help names ("--kokkos-num-threads=8"), each taking its value attached
        const std::string_view prefix = o.name.substr(0, o.name.size() - 1);
        std::istringstream words{std::string(o.help)};
        for (std::string w; words >> w;) {
            if (!w.starts_with(prefix) || w.find('*') != std::string::npos) continue;
            const size_t eq = w.find('=');
            const std::string name = w.substr(0, eq);
            if (std::any_of(out.begin(), out.end(), [&](const Flag & f) { return f.names[0] == name; })) continue;
            Flag f{{name}, eq != std::string::npos, eq != std::string::npos, {}, ""};
            out.push_back(f);
        }
        return;
    }
    Flag f;
    f.names.emplace_back(o.name);
    if (!o.alias.empty()) f.names.emplace_back(o.alias);
    f.takes_value = !o.value.empty();
    f.attached = o.value.starts_with('=');
    if (f.takes_value) f.values = values_of(o.value, o.name);
    f.help = brief(o.help);
    out.push_back(f);
}

std::vector<Node> nodes() {
    std::vector<Node> out;
    auto node = [&](const std::string & path) -> Node & {
        for (Node & n : out) {
            if (n.path == path) return n;
        }
        out.push_back({path, {}, {}, {}, false});
        return out.back();
    };
    auto add_child = [](Node & n, const std::string & word, std::string_view summary) {
        for (const auto & c : n.children) {
            if (c.first == word) return;
        }
        n.children.emplace_back(word, summary);
    };
    node("");
    for (const Command & c : commands()) {
        if (c.name.starts_with('-')) continue;
        const std::vector<std::string> w = split(c.name, ' ');
        std::string path;
        for (size_t i = 0; i < w.size() && !c.name.empty(); i++) {
            std::string_view summary = c.summary;
            if (i + 1 < w.size()) {
                for (const Family & f : families()) {
                    if (f.name == w[i]) summary = f.summary;
                }
            }
            add_child(node(path), w[i], brief(summary));
            path += (path.empty() ? "" : " ") + w[i];
        }
        Node & n = node(path);
        for (const Option & o : c.options) add_option(n.options, o);
        for (const std::string_view a : c.arguments) n.positional.push_back(values_of(a));
        n.variadic = !c.arguments.empty() && c.arguments.back().ends_with("...]");
    }
    Node & root = out.front();
    for (const Option & o : global_options()) add_option(root.options, o);
    for (const Command & c : commands()) {
        if (c.name.starts_with('-')) root.options.push_back({{std::string(c.name)}, false, false, {}, brief(c.summary)});
    }
    for (Node & n : out) n.options.push_back({{"--help", "-h"}, false, false, {}, "Show the page of this command"});
    return out;
}

std::string join(const std::vector<std::string> & words, std::string_view separator = " ") {
    std::string out;
    for (const std::string & w : words) out += (out.empty() ? "" : std::string(separator)) + w;
    return out;
}

/** @brief text in single quotes, for bash, zsh and fish alike. */
std::string quote(std::string_view text) {
    std::string out = "'";
    for (const char c : text) out += c == '\'' ? std::string("'\\''") : std::string(1, c);
    return out + "'";
}

std::string fish_quote(std::string_view text) {
    std::string out = "'";
    for (const char c : text) {
        if (c == '\'' || c == '\\') out += '\\';
        out += c;
    }
    return out + "'";
}

std::string option_word(const Flag & f, const std::string & name) { return f.attached ? name + "=" : name; }

std::string header() {
    return "# Completion for mallard " + mallard_version() + ", made by `mallard completion` from " +
           "its command table.\n";
}

/** @brief Shell case entries "<cmd> <option>" -> spec, specific commands first, then the global options. */
std::vector<std::pair<std::string, std::string>> value_entries(const std::vector<Node> & all) {
    std::vector<std::pair<std::string, std::string>> out;
    for (const Node & n : all) {
        for (const Flag & f : n.options) {
            if (!f.takes_value || f.attached) continue;
            for (const std::string & name : f.names) out.emplace_back(n.path + " " + name, join(f.values));
        }
    }
    return out;
}

/** @brief Entries "<cmd> <index>" -> spec; a variadic last argument as "<cmd> *". */
std::vector<std::pair<std::string, std::string>> positional_entries(const std::vector<Node> & all) {
    std::vector<std::pair<std::string, std::string>> out;
    for (const Node & n : all) {
        for (size_t i = 0; i < n.positional.size(); i++) {
            const bool last = n.variadic && i + 1 == n.positional.size();
            out.emplace_back(n.path + " " + (last ? "*" : std::to_string(i)), join(n.positional[i]));
        }
    }
    return out;
}

/** @brief A case pattern for "<cmd> <rest>": the quoted literal, with a trailing * unquoted. */
std::string pattern(const std::string & key) {
    return key.ends_with('*') ? quote(key.substr(0, key.size() - 1)) + "*" : quote(key);
}

/** @brief The global options' value entries, valid after any command. */
std::vector<std::pair<std::string, std::string>> global_value_entries() {
    std::vector<Flag> flags;
    for (const Option & o : global_options()) add_option(flags, o);
    std::vector<std::pair<std::string, std::string>> out;
    for (const Flag & f : flags) {
        if (f.takes_value && !f.attached) out.emplace_back("* " + f.names[0], join(f.values));
    }
    return out;
}

std::string global_pattern(const std::string & key) { return "*" + quote(key.substr(1)); }

constexpr std::string_view TEMPLATES = "awk '$2 ~ /^[23]D$/ {print $1}'";

std::string bash() {
    const std::vector<Node> all = nodes();
    std::ostringstream s;
    s << header() << "# Load it with: source <(mallard completion bash)\n\n";
    s << "_mallard_children() {\n    case $1 in\n";
    for (const Node & n : all) {
        std::vector<std::string> w;
        for (const auto & c : n.children) w.push_back(c.first);
        if (!w.empty()) s << "        " << quote(n.path) << ") REPLY=" << quote(join(w)) << " ;;\n";
    }
    s << "        *) REPLY='' ;;\n    esac\n}\n\n";
    s << "_mallard_options() {\n    case $1 in\n";
    for (const Node & n : all) {
        std::vector<std::string> w;
        for (const Flag & f : n.options) {
            for (const std::string & name : f.names) w.push_back(option_word(f, name));
        }
        s << "        " << quote(n.path) << ") REPLY=" << quote(join(w)) << " ;;\n";
    }
    s << "        *) REPLY='' ;;\n    esac\n}\n\n";
    s << "# What completes the value of option $2 of command $1; status 1 if it takes none\n";
    s << "_mallard_value() {\n    case \"$1 $2\" in\n";
    for (const auto & [key, spec] : value_entries(all)) s << "        " << pattern(key) << ") REPLY=" << quote(spec) << " ;;\n";
    for (const auto & [key, spec] : global_value_entries()) {
        s << "        " << global_pattern(key) << ") REPLY=" << quote(spec) << " ;;\n";
    }
    s << "        *) return 1 ;;\n    esac\n}\n\n";
    s << "# What completes positional argument $2 (from 0) of command $1\n";
    s << "_mallard_positional() {\n    case \"$1 $2\" in\n";
    for (const auto & [key, spec] : positional_entries(all)) {
        s << "        " << pattern(key) << ") REPLY=" << quote(spec) << " ;;\n";
    }
    s << "        *) return 1 ;;\n    esac\n}\n\n";
    s << R"SH(_mallard_add() {
    local IFS=' ' token words='' tokens
    read -ra tokens <<< "$1"
    IFS=$'\n'
    for token in "${tokens[@]}"; do
        case $token in
            @files) COMPREPLY+=($(compgen -f -- "$2")) ;;
            @files:*) COMPREPLY+=($(compgen -d -- "$2") $(compgen -f -X "!${token#@files:}" -- "$2")) ;;
            @dirs) COMPREPLY+=($(compgen -d -- "$2")) ;;
            @templates) COMPREPLY+=($(compgen -W "$("${COMP_WORDS[0]}" new 2>/dev/null | )SH"
      << TEMPLATES << R"SH()" -- "$2")) ;;
            *) words+=" $token" ;;
        esac
    done
    [[ -n $words ]] && COMPREPLY+=($(IFS=' '; compgen -W "$words" -- "$2"))
    return 0
}

_mallard() {
    local words=() cword=0 j REPLY
    # Rejoin what COMP_WORDBREAKS split at = and :
    for ((j = 0; j < ${#COMP_WORDS[@]}; j++)); do
        if ((j > 0 && ${#words[@]} > 0)) && [[ ${COMP_WORDS[j]} == [=:] || ${COMP_WORDS[j-1]} == [=:] ]]; then
            words[${#words[@]}-1]+=${COMP_WORDS[j]}
        else
            words+=("${COMP_WORDS[j]}")
        fi
        ((j == COMP_CWORD)) && cword=$((${#words[@]} - 1))
    done
    local cur=${words[cword]} cmd='' n=0 i w
    for ((i = 1; i < cword; i++)); do
        w=${words[i]}
        if [[ $w == -?* ]]; then
            [[ $w != *=* ]] && _mallard_value "$cmd" "$w" && ((i++))
            continue
        fi
        _mallard_children "$cmd"
        if ((n == 0)) && [[ " $REPLY " == *" $w "* ]]; then
            cmd=${cmd:+$cmd }$w
        else
            n=$((n + 1))
        fi
    done
    COMPREPLY=()
    if [[ $cur == -?*=* ]]; then
        if _mallard_value "$cmd" "${cur%%=*}"; then
            _mallard_add "$REPLY" "${cur#*=}"
            if [[ $COMP_WORDBREAKS != *=* ]]; then
                for ((j = 0; j < ${#COMPREPLY[@]}; j++)); do COMPREPLY[j]=${cur%%=*}=${COMPREPLY[j]}; done
            fi
        fi
        return 0
    fi
    local prev=${words[cword-1]}
    if ((cword > 1)) && [[ $prev == -?* && $prev != *=* ]] && _mallard_value "$cmd" "$prev"; then
        _mallard_add "$REPLY" "$cur"
        return 0
    fi
    if [[ $cur == -* ]]; then
        _mallard_options "$cmd"
        COMPREPLY=($(compgen -W "$REPLY" -- "$cur"))
        if ((${#COMPREPLY[@]} == 1)) && [[ ${COMPREPLY[0]} == *= ]]; then
            type compopt &>/dev/null && compopt -o nospace
        fi
        return 0
    fi
    _mallard_children "$cmd"
    ((n == 0)) && [[ -n $REPLY ]] && _mallard_add "$REPLY" "$cur"
    _mallard_positional "$cmd" "$n" && _mallard_add "$REPLY" "$cur"
    return 0
}

complete -o filenames -F _mallard mallard mallard2d mallard3d
)SH";
    return s.str();
}

/** @brief "word:description" for zsh's _describe, the word's colons escaped. */
std::string described(const std::string & word, std::string_view help) {
    std::string w;
    for (const char c : word) w += c == ':' ? std::string("\\:") : std::string(1, c);
    return quote(help.empty() ? w : w + ":" + std::string(help));
}

/** @brief Each word of a spec in single quotes, for a zsh array. */
std::string quoted(const std::string & spec) {
    std::vector<std::string> out;
    std::istringstream words(spec);
    for (std::string w; words >> w;) out.push_back(quote(w));
    return join(out);
}

std::string zsh() {
    const std::vector<Node> all = nodes();
    std::ostringstream s;
    s << "#compdef mallard mallard2d mallard3d\n" << header() << "\n";
    s << "_mallard_children() {\n    case $1 in\n";
    for (const Node & n : all) {
        if (n.children.empty()) continue;
        std::vector<std::string> w;
        for (const auto & c : n.children) w.push_back(described(c.first, c.second));
        s << "        (" << quote(n.path) << ") reply=(" << join(w) << ") ;;\n";
    }
    s << "        (*) reply=() ;;\n    esac\n}\n\n";
    s << "_mallard_options() {\n    case $1 in\n";
    for (const Node & n : all) {
        std::vector<std::string> w, attached;
        for (const Flag & f : n.options) {
            for (const std::string & name : f.names) {
                (f.attached ? attached : w).push_back(described(option_word(f, name), f.help));
            }
        }
        s << "        (" << quote(n.path) << ") reply=(" << join(w) << ") eq=(" << join(attached) << ") ;;\n";
    }
    s << "        (*) reply=() eq=() ;;\n    esac\n}\n\n";
    s << "_mallard_value() {\n    case \"$1 $2\" in\n";
    for (const auto & [key, spec] : value_entries(all)) {
        s << "        (" << pattern(key) << ") reply=(" << quoted(spec) << ") ;;\n";
    }
    for (const auto & [key, spec] : global_value_entries()) {
        s << "        (" << global_pattern(key) << ") reply=(" << quoted(spec) << ") ;;\n";
    }
    s << "        (*) return 1 ;;\n    esac\n}\n\n";
    s << "_mallard_positional() {\n    case \"$1 $2\" in\n";
    for (const auto & [key, spec] : positional_entries(all)) {
        s << "        (" << pattern(key) << ") reply=(" << quoted(spec) << ") ;;\n";
    }
    s << "        (*) return 1 ;;\n    esac\n}\n\n";
    s << R"SH(_mallard_add() {
    local token
    local -a lits
    for token in "$@"; do
        case $token in
            (@files) _files ;;
            (@files:*) _files -g "${token#@files:}" ;;
            (@dirs) _files -/ ;;
            (@templates) lits+=(${(f)"$(${(Q)words[1]} new 2>/dev/null | )SH"
      << TEMPLATES << R"SH()"}) ;;
            (*) lits+=("$token") ;;
        esac
    done
    (( $#lits )) && compadd -a lits
    return 0
}

_mallard() {
    local cmd='' w prev
    local -i i n=0
    local -a reply subs opts eq
    for ((i = 2; i < CURRENT; i++)); do
        w=${words[i]}
        if [[ $w == -?* ]]; then
            [[ $w != *=* ]] && _mallard_value "$cmd" "$w" && ((i++))
            continue
        fi
        _mallard_children "$cmd"
        if ((n == 0)) && ((${reply[(I)${(b)w}(|:*)]})); then
            cmd=${cmd:+$cmd }$w
        else
            n+=1
        fi
    done
    if [[ $PREFIX == -?*=* ]]; then
        if _mallard_value "$cmd" "${PREFIX%%=*}"; then
            compset -P '*='
            _mallard_add "${reply[@]}"
        fi
        return 0
    fi
    prev=${words[CURRENT-1]}
    if ((CURRENT > 2)) && [[ $prev == -?* && $prev != *=* ]] && _mallard_value "$cmd" "$prev"; then
        _mallard_add "${reply[@]}"
        return 0
    fi
    if [[ $PREFIX == -* ]]; then
        _mallard_options "$cmd"
        opts=("${reply[@]}")
        _describe -t options option opts
        (( $#eq )) && _describe -t options option eq -S ''
        return 0
    fi
    if ((n == 0)); then
        _mallard_children "$cmd"
        subs=("${reply[@]}")
        (( $#subs )) && _describe -t commands command subs
    fi
    _mallard_positional "$cmd" "$n" && _mallard_add "${reply[@]}"
    return 0
}

if [[ $funcstack[1] == _mallard ]]; then
    _mallard "$@"
else
    compdef _mallard mallard mallard2d mallard3d
fi
)SH";
    return s.str();
}

std::string fish_args(const Values & v, bool & files) {
    std::vector<std::string> words;
    std::string out;
    for (const std::string & t : v) {
        if (t == "@files") {
            files = true;
        } else if (t.starts_with("@files:")) {
            out += " -a " + fish_quote("(__fish_complete_suffix " + t.substr(t.find('*') + 1) + ")");
        } else if (t == "@dirs") {
            out += " -a '(__fish_complete_directories)'";
        } else if (t == "@templates") {
            out += " -a '(__mallard_templates)'";
        } else {
            words.push_back(t);
        }
    }
    if (!words.empty()) out += " -a " + fish_quote(join(words));
    return out;
}

std::string fish() {
    const std::vector<Node> all = nodes();
    std::ostringstream s;
    s << header() << "\n";
    s << "function __mallard_children\n    switch $argv[1]\n";
    for (const Node & n : all) {
        if (n.children.empty()) continue;
        std::vector<std::string> w;
        for (const auto & c : n.children) w.push_back(c.first);
        s << "        case " << fish_quote(n.path) << "\n            printf '%s\\n' " << join(w) << "\n";
    }
    s << "    end\nend\n\n";
    s << "function __mallard_takes_value\n    switch \"$argv[1] $argv[2]\"\n";
    std::vector<std::string> keys;
    for (const auto & [key, spec] : value_entries(all)) keys.push_back(fish_quote(key));
    for (const auto & [key, spec] : global_value_entries()) keys.push_back(fish_quote(key));
    s << "        case " << join(keys) << "\n            return 0\n    end\n    return 1\nend\n\n";
    s << R"SH(# The command the words before the cursor name, then how many positional arguments follow it
function __mallard_state
    set -l tokens (commandline -xpc 2>/dev/null; or commandline -opc)
    set -e tokens[1]
    set -l cmd ''
    set -l n 0
    set -l skip 0
    for w in $tokens
        if test $skip = 1
            set skip 0
            continue
        end
        if string match -qr -- '^-.' $w
            if not string match -q -- '*=*' $w; and __mallard_takes_value "$cmd" $w
                set skip 1
            end
            continue
        end
        if test $n = 0; and contains -- $w (__mallard_children "$cmd")
            set cmd (string trim -- "$cmd $w")
        else
            set n (math $n + 1)
        end
    end
    printf '%s\n' "$cmd" $n
end

function __mallard_is
    set -l state (__mallard_state)
    test "$state[1]" = "$argv[1]"
end

# Positional argument $argv[2] of command $argv[1]; with a third argument, any from $argv[2] on
function __mallard_arg
    set -l state (__mallard_state)
    test "$state[1]" = "$argv[1]"; or return 1
    test "$state[2]" = "$argv[2]"; and return 0
    test (count $argv) -ge 3; and test "$state[2]" -ge "$argv[2]"
end

function __mallard_templates
    set -l program (commandline -xpc 2>/dev/null; or commandline -opc)[1]
    $program new 2>/dev/null | )SH"
      << TEMPLATES << R"SH(
end

complete -c mallard -f
complete -c mallard2d -w mallard
complete -c mallard3d -w mallard
)SH";
    for (const Node & n : all) {
        const std::string cond = fish_quote("__mallard_is " + fish_quote(n.path));
        for (const Flag & f : n.options) {
            std::string line = "complete -c mallard -n " + cond;
            for (const std::string & name : f.names) {
                if (name.starts_with("--")) {
                    line += " -l " + name.substr(2);
                } else if (name.size() == 2) {
                    line += " -s " + name.substr(1);
                } else {
                    line += " -o " + name.substr(1);
                }
            }
            if (f.takes_value) {
                bool files = false;
                line += fish_args(f.values, files);
                line += files ? " -r -F" : " -x";
            }
            if (!f.help.empty()) line += " -d " + fish_quote(f.help);
            s << line << "\n";
        }
        for (const auto & [word, summary] : n.children) {
            s << "complete -c mallard -n " << fish_quote("__mallard_arg " + fish_quote(n.path) + " 0") << " -a "
              << fish_quote(word) << " -d " << fish_quote(summary) << "\n";
        }
        for (size_t i = 0; i < n.positional.size(); i++) {
            const bool last = n.variadic && i + 1 == n.positional.size();
            bool files = false;
            const std::string args = fish_args(n.positional[i], files);
            if (args.empty() && !files) continue;
            s << "complete -c mallard -n "
              << fish_quote("__mallard_arg " + fish_quote(n.path) + " " + std::to_string(i) + (last ? " +" : ""))
              << args << (files ? " -F" : "") << "\n";
        }
    }
    return s.str();
}

} // namespace

std::string completion_script(std::string_view shell) {
    if (shell == "bash") return bash();
    if (shell == "zsh") return zsh();
    if (shell == "fish") return fish();
    return "";
}

} // namespace cli
