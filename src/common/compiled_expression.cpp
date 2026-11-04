/**
 * @file compiled_expression.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Parser of compiled expressions: exprtk's grammar and precedences
 *        (precedence climbing) and its statements, constants folded, emitted
 *        as a stack program with jumps.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "compiled_expression.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>

#include <exprtk.hpp>

#include "units.h"

namespace expr {

namespace {

enum class Tok { NUMBER, NAME, OP, LPAREN, RPAREN, COMMA, QUESTION, COLON, SEMICOLON, TILDE, END };

struct Token {
    Tok type;
    std::string text;
    size_t pos;
    bool implicit = false;  // a '*' of implicit multiplication
};

std::string lower(std::string s) {
    for (char & c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

bool is_name_start(const char c) { return std::isalpha(static_cast<unsigned char>(c)) || c == '_'; }
bool is_name_char(const char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; }

bool is_keyword_operator(const std::string & name) {
    static const std::vector<std::string> words = {"and", "or", "xor", "nand", "nor", "xnor"};
    return std::find(words.begin(), words.end(), lower(name)) != words.end();
}

bool is_keyword(const std::string & name) {
    static const std::vector<std::string> words = {"if", "else", "while", "for", "repeat", "until", "switch",
                                                   "case", "default", "var", "break", "continue", "return"};
    return std::find(words.begin(), words.end(), lower(name)) != words.end();
}

bool is_constant_name(const std::string & name) {
    static const std::vector<std::string> words = {"pi", "epsilon", "inf", "true", "false"};
    return std::find(words.begin(), words.end(), lower(name)) != words.end();
}

struct Function {
    Op op;
    int min_args, max_args;  // max_args < 0: variadic
};

const std::unordered_map<std::string, Function> & functions() {
    static const std::unordered_map<std::string, Function> table = {
        {"abs", {Op::ABS, 1, 1}}, {"acos", {Op::ACOS, 1, 1}}, {"acosh", {Op::ACOSH, 1, 1}},
        {"asin", {Op::ASIN, 1, 1}}, {"asinh", {Op::ASINH, 1, 1}}, {"atan", {Op::ATAN, 1, 1}},
        {"atanh", {Op::ATANH, 1, 1}}, {"ceil", {Op::CEIL, 1, 1}}, {"cos", {Op::COS, 1, 1}},
        {"cosh", {Op::COSH, 1, 1}}, {"cot", {Op::COT, 1, 1}}, {"csc", {Op::CSC, 1, 1}},
        {"sec", {Op::SEC, 1, 1}}, {"erf", {Op::ERF, 1, 1}}, {"erfc", {Op::ERFC, 1, 1}},
        {"exp", {Op::EXP, 1, 1}}, {"expm1", {Op::EXPM1, 1, 1}}, {"floor", {Op::FLOOR, 1, 1}},
        {"frac", {Op::FRAC, 1, 1}}, {"log", {Op::LOG, 1, 1}}, {"log10", {Op::LOG10, 1, 1}},
        {"log1p", {Op::LOG1P, 1, 1}}, {"log2", {Op::LOG2, 1, 1}}, {"ncdf", {Op::NCDF, 1, 1}},
        {"round", {Op::ROUND, 1, 1}}, {"sgn", {Op::SGN, 1, 1}}, {"sin", {Op::SIN, 1, 1}},
        {"sinc", {Op::SINC, 1, 1}}, {"sinh", {Op::SINH, 1, 1}}, {"sqrt", {Op::SQRT, 1, 1}},
        {"tan", {Op::TAN, 1, 1}}, {"tanh", {Op::TANH, 1, 1}}, {"trunc", {Op::TRUNC, 1, 1}},
        {"deg2rad", {Op::DEG2RAD, 1, 1}}, {"rad2deg", {Op::RAD2DEG, 1, 1}}, {"not", {Op::NOT, 1, 1}},
        {"atan2", {Op::ATAN2, 2, 2}}, {"hypot", {Op::HYPOT, 2, 2}}, {"logn", {Op::LOGN, 2, 2}},
        {"root", {Op::ROOT, 2, 2}}, {"roundn", {Op::ROUNDN, 2, 2}}, {"pow", {Op::POW, 2, 2}},
        {"clamp", {Op::CLAMP, 3, 3}}, {"inrange", {Op::INRANGE, 3, 3}},
        {"min", {Op::MIN, 1, -1}}, {"max", {Op::MAX, 1, -1}}, {"sum", {Op::ADD, 1, -1}},
        {"mul", {Op::MUL, 1, -1}}, {"avg", {Op::DIV, 1, -1}},
    };
    return table;
}

/** @brief What a node computes: an operation on its arguments, or a statement. */
enum class Kind { OP, LOCAL, ASSIGN, SEQUENCE, IF, WHILE, FOR, REPEAT, BREAK, CONTINUE };

/**
 * @brief A node of the syntax tree. Arguments: OP its operands; ASSIGN the
 *        value; SEQUENCE the statements; IF condition, consequent,
 *        alternative; WHILE condition, body; FOR initializer, condition,
 *        incrementer (or null), body; REPEAT body, condition; BREAK its value
 *        (or none).
 */
struct Node {
    Kind kind = Kind::OP;
    Op op = Op::CONST;
    double value = 0.0;
    uint16_t index = 0;  // OP: as Instr::index; LOCAL, ASSIGN: the slot
    std::vector<std::unique_ptr<Node>> args;
};
using NodePtr = std::unique_ptr<Node>;

NodePtr constant(const double v) {
    auto n = std::make_unique<Node>();
    n->value = v;
    return n;
}

bool is_constant(const Node & n) { return n.kind == Kind::OP && n.op == Op::CONST; }

bool all_constant(const Node & n) {
    for (const auto & a : n.args) {
        if (!is_constant(*a)) return false;
    }
    return true;
}

/** @brief Whether evaluating n can change anything but its value (assignments, loops, break, continue). */
bool has_effects(const Node * n) {
    if (!n) return false;
    if (n->kind != Kind::OP && n->kind != Kind::LOCAL && n->kind != Kind::SEQUENCE && n->kind != Kind::IF) {
        return true;
    }
    return std::any_of(n->args.begin(), n->args.end(), [](const auto & a) { return has_effects(a.get()); });
}

double evaluate_constant(const Node & n) {
    std::vector<Instr> code;
    for (const auto & a : n.args) code.push_back(Instr{a->value, 0, Op::CONST});
    code.push_back(Instr{0.0, n.index, n.op});
    return evaluate(code.data(), static_cast<uint32_t>(code.size()), ArrayVariables{nullptr});
}

NodePtr node(const Kind kind, std::vector<NodePtr> args) {
    auto n = std::make_unique<Node>();
    n->kind = kind;
    n->args = std::move(args);
    return n;
}

template <typename... Args>
std::vector<NodePtr> list(Args &&... args) {
    std::vector<NodePtr> v;
    (v.push_back(std::forward<Args>(args)), ...);
    return v;
}

/** @brief The node op(args), with exprtk's integer powers and constants folded. */
NodePtr make(const Op op, std::vector<NodePtr> args) {
    auto n = node(Kind::OP, std::move(args));
    n->op = op;
    if (op == Op::POW && is_constant(*n->args[1])) {
        const double c = n->args[1]->value;
        if (std::abs(c) <= 60.0 && c == std::trunc(c)) {
            const uint16_t p = static_cast<uint16_t>(std::abs(c));
            if (p == 0) return constant(1.0);
            n->op = c >= 0.0 ? Op::IPOW : Op::IPOWINV;
            n->index = p;
            n->args.pop_back();
        }
    }
    if (op == Op::NEG && n->args[0]->kind == Kind::OP && n->args[0]->op == Op::NEG) {
        return std::move(n->args[0]->args[0]);
    }
    if (all_constant(*n)) return constant(evaluate_constant(*n));
    return n;
}

NodePtr make(const Op op, NodePtr a) { return make(op, list(std::move(a))); }

NodePtr make(const Op op, NodePtr a, NodePtr b) { return make(op, list(std::move(a), std::move(b))); }

/**
 * @brief exprtk's conditional: c ? a : b, b NaN if absent. Without effects
 *        in the branches, the select of both (as exprtk's ternary operator).
 */
NodePtr conditional(NodePtr c, NodePtr a, NodePtr b) {
    if (!b) b = constant(std::numeric_limits<double>::quiet_NaN());
    if (!has_effects(a.get()) && !has_effects(b.get())) {
        return make(Op::SELECT, list(std::move(c), std::move(a), std::move(b)));
    }
    if (is_constant(*c)) return c->value != 0.0 ? std::move(a) : std::move(b);
    return node(Kind::IF, list(std::move(c), std::move(a), std::move(b)));
}

/**
 * @brief Statements evaluated in order, valued as the last, as exprtk keeps
 *        them: statements without effects dropped unless last, and none
 *        after a break or continue.
 */
NodePtr sequence(std::vector<NodePtr> statements) {
    if (statements.empty()) return constant(std::numeric_limits<double>::quiet_NaN());
    std::vector<NodePtr> kept;
    for (size_t i = 0; i < statements.size(); i++) {
        const Kind k = statements[i]->kind;
        const bool last = i + 1 == statements.size();
        if (last || has_effects(statements[i].get())) kept.push_back(std::move(statements[i]));
        if (k == Kind::BREAK || k == Kind::CONTINUE) break;
    }
    if (kept.size() == 1) return std::move(kept[0]);
    return node(Kind::SEQUENCE, std::move(kept));
}

/** @brief exprtk's value of a numeric literal (its own conversion, not strtod). */
double literal_value(const std::string & text) {
    double v = 0.0;
    if (!exprtk::details::string_to_real(text, v)) throw std::invalid_argument("invalid number " + text);
    return v;
}

class Parser {
    public:
        Parser(const std::string & name_, const std::string & text_, const std::vector<std::string> & variables_) :
            name(name_), text(text_), variables(variables_) {
            tokenize();
        }

        NodePtr parse() {
            NodePtr e = statements([] { return false; }, Tok::SEMICOLON, "';'");
            if (peek().type != Tok::END) fail("unexpected '" + peek().text + "'", peek().pos);
            return e;
        }

        uint16_t n_locals() const { return max_live; }

        [[noreturn]] void fail(const std::string & what, const size_t pos) const {
            throw std::invalid_argument("Failed to parse expression for " + name + " (\"" + text + "\"): " + what +
                                        " at position " + std::to_string(pos + 1) + ".");
        }

    private:
        const std::string & name;
        const std::string & text;
        const std::vector<std::string> & variables;
        std::vector<Token> tokens;
        size_t at = 0;
        std::vector<std::vector<std::pair<std::string, uint16_t>>> scopes{1};
        uint16_t live = 0, max_live = 0;
        std::vector<bool> loop_breaks;  // per enclosing loop: whether a break leaves it

        const Token & peek() const { return tokens[at]; }
        Token next() { return tokens[at++]; }

        void expect(const Tok type, const std::string & what) {
            if (peek().type != type) fail("expected " + what, peek().pos);
            at++;
        }

        void expect_text(const std::string & t, const std::string & what) {
            if (peek().text != t) fail("expected " + what, peek().pos);
            at++;
        }

        static bool is_word(const Token & t, const char * word) {
            return t.type == Tok::NAME && lower(t.text) == word;
        }

        bool is_function(const Token & t) const {
            return t.type == Tok::NAME && functions().count(lower(t.text)) > 0;
        }

        bool is_variable(const Token & t) const {
            return t.type == Tok::NAME && std::find(variables.begin(), variables.end(), t.text) != variables.end();
        }

        void tokenize() {
            size_t i = 0;
            while (i < text.size()) {
                const char c = text[i];
                if (std::isspace(static_cast<unsigned char>(c))) {
                    i++;
                    continue;
                }
                const size_t start = i;
                if (std::isdigit(static_cast<unsigned char>(c)) ||
                    (c == '.' && i + 1 < text.size() && std::isdigit(static_cast<unsigned char>(text[i + 1])))) {
                    while (i < text.size() && (std::isdigit(static_cast<unsigned char>(text[i])) || text[i] == '.')) i++;
                    if (i < text.size() && (text[i] == 'e' || text[i] == 'E')) {
                        size_t j = i + 1;
                        if (j < text.size() && (text[j] == '+' || text[j] == '-')) j++;
                        if (j < text.size() && std::isdigit(static_cast<unsigned char>(text[j]))) {
                            i = j;
                            while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i]))) i++;
                        }
                    }
                    push(Tok::NUMBER, text.substr(start, i - start), start);
                } else if (is_name_start(c)) {
                    while (i < text.size() && is_name_char(text[i])) i++;
                    push(Tok::NAME, text.substr(start, i - start), start);
                } else if (c == '(' || c == '[' || c == '{') {
                    push(Tok::LPAREN, std::string(1, c), start);
                    i++;
                } else if (c == ')' || c == ']' || c == '}') {
                    push(Tok::RPAREN, std::string(1, c), start);
                    i++;
                } else if (c == ',') {
                    push(Tok::COMMA, ",", start);
                    i++;
                } else if (c == '?') {
                    push(Tok::QUESTION, "?", start);
                    i++;
                } else if (c == ';') {
                    push(Tok::SEMICOLON, ";", start);
                    i++;
                } else if (c == '~') {
                    push(Tok::TILDE, "~", start);
                    i++;
                } else if (c == ':' && !(i + 1 < text.size() && text[i + 1] == '=')) {
                    push(Tok::COLON, ":", start);
                    i++;
                } else if (c == '\'') {
                    fail("strings are not supported", start);
                } else {
                    static const std::vector<std::string> ops = {":=", "+=", "-=", "*=", "/=", "%=", "<=", ">=",
                                                                 "==", "!=", "<>", "<",  ">",  "=",  "+",  "-",
                                                                 "*",  "/",  "%",  "^",  "&",  "|"};
                    bool found = false;
                    for (const auto & op : ops) {
                        if (text.compare(i, op.size(), op) == 0) {
                            push(Tok::OP, op, start);
                            i += op.size();
                            found = true;
                            break;
                        }
                    }
                    if (!found) fail(std::string("unsupported character '") + c + "'", start);
                }
            }
            tokens.push_back(Token{Tok::END, "end of expression", text.size()});
        }

        // exprtk's implicit multiplication: 2x, 2(x), (x)(y), (x)y, x(y)
        void push(const Tok type, const std::string & t, const size_t pos) {
            if (!tokens.empty()) {
                const Token & prev = tokens.back();
                const bool name_like = type == Tok::NAME && !is_keyword_operator(t) && !is_keyword(t);
                const bool open = type == Tok::LPAREN;
                const bool implicit = (prev.type == Tok::NUMBER && (name_like || open)) ||
                                      (prev.type == Tok::RPAREN && (name_like || open || type == Tok::NUMBER)) ||
                                      (prev.type == Tok::NAME && open && !is_function(prev) && is_variable(prev));
                if (implicit) tokens.push_back(Token{Tok::OP, "*", pos, true});
            }
            tokens.push_back(Token{type, t, pos});
        }

        /** @brief exprtk's binary operators: (left precedence, right precedence, op), or false. */
        bool binary_operator(const Token & t, int & left, int & right, Op & op) const {
            const std::string s = t.type == Tok::NAME ? lower(t.text) : t.text;
            if (t.type != Tok::OP && !(t.type == Tok::NAME && is_keyword_operator(s))) return false;
            struct Entry {
                const char * s;
                int l, r;
                Op op;
            };
            static const Entry table[] = {
                {"or", 1, 2, Op::OR},    {"|", 1, 2, Op::OR},     {"nor", 1, 2, Op::NOR},  {"xor", 1, 2, Op::XOR},
                {"xnor", 1, 2, Op::XNOR}, {"and", 3, 4, Op::AND}, {"&", 3, 4, Op::AND},    {"nand", 3, 4, Op::NAND},
                {"<", 5, 6, Op::LT},     {"<=", 5, 6, Op::LE},    {">", 5, 6, Op::GT},     {">=", 5, 6, Op::GE},
                {"==", 5, 6, Op::EQ},    {"=", 5, 6, Op::EQ},     {"!=", 5, 6, Op::NE},    {"<>", 5, 6, Op::NE},
                {"+", 7, 8, Op::ADD},    {"-", 7, 8, Op::SUB},    {"*", 10, 11, Op::MUL},  {"/", 10, 11, Op::DIV},
                {"%", 10, 11, Op::MOD},  {"^", 12, 12, Op::POW},
            };
            for (const auto & e : table) {
                if (s == e.s) {
                    left = e.l;
                    right = e.r;
                    op = e.op;
                    return true;
                }
            }
            return false;
        }

        /** @brief exprtk's assignments (precedence 0, right-associative): CONST for :=, else the update's op. */
        static bool assignment_operator(const Token & t, Op & op) {
            if (t.type != Tok::OP) return false;
            static const std::pair<const char *, Op> table[] = {
                {":=", Op::CONST}, {"+=", Op::ADD}, {"-=", Op::SUB}, {"*=", Op::MUL}, {"/=", Op::DIV}, {"%=", Op::MOD},
            };
            for (const auto & [s, o] : table) {
                if (t.text == s) {
                    op = o;
                    return true;
                }
            }
            return false;
        }

        /**
         * @brief Statements separated by separator up to closed() (or the
         *        end); an if-else that took its own ';' needs no separator.
         */
        NodePtr statements(const std::function<bool()> & closed, const Tok separator, const std::string & what) {
            std::vector<NodePtr> list;
            for (;;) {
                if (closed() || peek().type == Tok::END) break;
                list.push_back(expression(0));
                if (closed() || peek().type == Tok::END) break;
                if (peek().type == separator) {
                    next();
                    continue;
                }
                if (separator == Tok::SEMICOLON && tokens[at - 1].type == Tok::SEMICOLON) continue;
                fail("expected " + what, peek().pos);
            }
            return sequence(std::move(list));
        }

        void open_scope() { scopes.emplace_back(); }

        void close_scope() {
            live = static_cast<uint16_t>(live - scopes.back().size());
            scopes.pop_back();
        }

        const std::pair<std::string, uint16_t> * local(const std::string & n) const {
            for (auto s = scopes.rbegin(); s != scopes.rend(); ++s) {
                for (const auto & l : *s) {
                    if (l.first == n) return &l;
                }
            }
            return nullptr;
        }

        uint16_t declare(const Token & t) {
            const std::string l = lower(t.text);
            if (is_keyword(t.text) || is_keyword_operator(t.text) || is_constant_name(t.text) ||
                functions().count(l) > 0) {
                fail("'" + t.text + "' is a reserved word and cannot name a variable", t.pos);
            }
            for (const auto & v : variables) {
                if (lower(v) == l) fail("variable '" + t.text + "' would hide '" + v + "'", t.pos);
            }
            for (const auto & s : scopes) {
                for (const auto & v : s) {
                    if (lower(v.first) == l) fail("variable '" + t.text + "' is already defined", t.pos);
                }
            }
            if (live >= MAX_STACK) fail("too many local variables", t.pos);
            scopes.back().emplace_back(t.text, live);
            max_live = std::max(max_live, ++live);
            return scopes.back().back().second;
        }

        NodePtr expression(const int precedence) {
            NodePtr e = branch();
            for (;;) {
                Op op = Op::CONST;
                if (assignment_operator(peek(), op)) {
                    if (precedence > 0) break;
                    const Token t = next();
                    e = assign(std::move(e), op, expression(0), t);
                    continue;
                }
                int left = 0, right = 0;
                if (!binary_operator(peek(), left, right, op) || left < precedence) break;
                next();
                e = make(op, std::move(e), expression(right));
            }
            if (precedence == 0 && peek().type == Tok::QUESTION) {
                next();
                NodePtr a = expression(0);
                expect(Tok::COLON, "':' of the ternary operator");
                NodePtr b = expression(0);
                e = conditional(std::move(e), std::move(a), std::move(b));
            }
            return e;
        }

        NodePtr assign(NodePtr target, const Op op, NodePtr value, const Token & t) {
            if (target->kind != Kind::LOCAL) {
                if (target->kind == Kind::OP && target->op == Op::VAR) {
                    const std::string & v = variables[target->index];
                    fail("cannot assign to the input variable '" + v + "'; copy it into a local variable (var v := " +
                             v + ")",
                         t.pos);
                }
                fail("'" + t.text + "' assigns only to local variables (declared with var)", t.pos);
            }
            const uint16_t slot = target->index;
            if (op != Op::CONST) value = make(op, std::move(target), std::move(value));
            auto n = node(Kind::ASSIGN, list(std::move(value)));
            n->index = slot;
            return n;
        }

        NodePtr branch() {
            const Token t = next();
            switch (t.type) {
                case Tok::NUMBER:
                    try {
                        return constant(literal_value(t.text));
                    } catch (const std::invalid_argument &) {
                        fail("invalid number '" + t.text + "'", t.pos);
                    }
                case Tok::LPAREN: {
                    NodePtr e = expression(0);
                    expect(Tok::RPAREN, "')'");
                    return e;
                }
                case Tok::OP:
                    if (t.text == "-") return make(Op::NEG, expression(11));
                    if (t.text == "+") return expression(13);
                    fail("unexpected '" + t.text + "'", t.pos);
                case Tok::TILDE:
                    if (peek().text == "{") return block("~");
                    if (peek().text == "(") {
                        next();
                        open_scope();
                        NodePtr e = statements([this] { return peek().type == Tok::RPAREN; }, Tok::COMMA, "','");
                        close_scope();
                        expect_text(")", "')' after the statements of ~(...)");
                        return e;
                    }
                    fail("expected '{' or '(' after '~'", peek().pos);
                case Tok::NAME:
                    return is_keyword(t.text) ? statement(t) : symbol(t);
                default:
                    fail("unexpected " + (t.type == Tok::END ? t.text : "'" + t.text + "'"), t.pos);
            }
        }

        /** @brief { statements }, a scope of its own; as exprtk, an implicit '*' before it is skipped. */
        NodePtr block(const std::string & what) {
            if (peek().implicit) next();
            if (peek().text != "{") fail("expected '{' to begin the body of " + what, peek().pos);
            next();
            open_scope();
            NodePtr e = statements([this] { return peek().text == "}"; }, Tok::SEMICOLON, "';'");
            close_scope();
            expect_text("}", "'}' to end the body of " + what);
            return e;
        }

        void open_condition(const std::string & what) {
            if (peek().text != "(") fail("expected '(' after " + what, peek().pos);
            next();
        }

        NodePtr loop_body(const std::string & what, bool & breaks) {
            loop_breaks.push_back(false);
            NodePtr body = block(what);
            breaks = loop_breaks.back();
            loop_breaks.pop_back();
            return body;
        }

        NodePtr statement(const Token & t) {
            const std::string k = lower(t.text);
            if (k == "var") return declaration();
            if (k == "if") return if_statement();
            if (k == "while") {
                open_condition("while");
                NodePtr c = expression(0);
                expect_text(")", "')' after the condition of while");
                bool breaks = false;
                NodePtr body = loop_body("while", breaks);
                if (is_constant(*c) && c->value != 0.0 && !breaks) {
                    fail("the condition of this while loop is always true and its body has no break", t.pos);
                }
                return node(Kind::WHILE, list(std::move(c), std::move(body)));
            }
            if (k == "for") {
                open_condition("for");
                open_scope();
                NodePtr init = expression(0);
                expect(Tok::SEMICOLON, "';' after the initializer of for");
                NodePtr c = expression(0);
                expect(Tok::SEMICOLON, "';' after the condition of for");
                NodePtr step = peek().type == Tok::RPAREN ? nullptr : expression(0);
                expect_text(")", "')' after the incrementer of for");
                bool breaks = false;
                NodePtr body = loop_body("for", breaks);
                close_scope();
                if (is_constant(*c) && c->value != 0.0 && !breaks) {
                    fail("the condition of this for loop is always true and its body has no break", t.pos);
                }
                return node(Kind::FOR, list(std::move(init), std::move(c), std::move(step), std::move(body)));
            }
            if (k == "repeat") {
                open_scope();
                loop_breaks.push_back(false);
                NodePtr body = statements([this] { return is_word(peek(), "until"); }, Tok::SEMICOLON, "';'");
                const bool breaks = loop_breaks.back();
                loop_breaks.pop_back();
                close_scope();
                if (!is_word(peek(), "until")) fail("expected until to end the repeat loop", peek().pos);
                next();
                open_condition("until");
                NodePtr c = expression(0);
                expect_text(")", "')' after the condition of until");
                if (is_constant(*c) && c->value == 0.0 && !breaks) {
                    fail("the condition of this repeat loop is always false and its body has no break", t.pos);
                }
                return node(Kind::REPEAT, list(std::move(body), std::move(c)));
            }
            if (k == "switch") return switch_statement();
            if (k == "break" || k == "continue") {
                if (loop_breaks.empty()) fail(k + " outside a loop", t.pos);
                if (k == "continue") return node(Kind::CONTINUE, {});
                loop_breaks.back() = true;
                std::vector<NodePtr> value;
                if (peek().text == "[") {
                    next();
                    value.push_back(expression(0));
                    expect_text("]", "']' after the value of break");
                }
                return node(Kind::BREAK, std::move(value));
            }
            if (k == "return") fail("return is not supported", t.pos);
            fail("unexpected '" + t.text + "'", t.pos);
        }

        NodePtr declaration() {
            const Token v = next();
            if (v.type != Tok::NAME) fail("expected a variable name after var", v.pos);
            if (peek().text == "[") fail("vectors are not supported", peek().pos);
            NodePtr value;
            if (peek().type == Tok::OP && peek().text == ":=") {
                next();
                value = expression(0);
            } else {
                value = constant(0.0);
            }
            auto n = node(Kind::ASSIGN, list(std::move(value)));
            n->index = declare(v);
            return n;
        }

        // exprtk's forms: if (c, a, b); if (c) a; [else b;]; if (c) { ... } [else { ... }]; else if ...
        NodePtr if_statement() {
            open_condition("if");
            NodePtr c = expression(0);
            if (peek().type == Tok::COMMA) {
                next();
                NodePtr a = expression(0);
                expect(Tok::COMMA, "',' between the consequent and the alternative of if");
                NodePtr b = expression(0);
                expect_text(")", "')' after the alternative of if");
                return conditional(std::move(c), std::move(a), std::move(b));
            }
            expect_text(")", "')' or ',' after the condition of if");
            if (peek().implicit) next();
            NodePtr a;
            if (peek().text == "{") {
                a = block("if");
            } else {
                a = expression(0);
                if (peek().type != Tok::SEMICOLON && peek().type != Tok::END) {
                    fail("expected ';' after the consequent of if", peek().pos);
                }
            }
            bool has_else = is_word(peek(), "else");
            if (!has_else && peek().type == Tok::SEMICOLON && is_word(tokens[at + 1], "else")) {
                next();
                has_else = true;
            }
            if (!has_else) return conditional(std::move(c), std::move(a), nullptr);
            next();
            NodePtr b;
            if (peek().text == "{") {
                b = block("else");
            } else if (is_word(peek(), "if")) {
                next();
                b = if_statement();
            } else {
                b = expression(0);
                if (peek().type == Tok::SEMICOLON) next();
            }
            return conditional(std::move(c), std::move(a), std::move(b));
        }

        // switch { case c0 : v0; case c1 : v1; ... default : v; }: the value of the first true case, else the default
        NodePtr switch_statement() {
            if (peek().text != "{") fail("expected '{' after switch", peek().pos);
            next();
            std::vector<std::pair<NodePtr, NodePtr>> cases;
            NodePtr fallback;
            while (peek().text != "}") {
                const Token k = next();
                NodePtr c;
                if (is_word(k, "case")) {
                    c = expression(0);
                } else if (!is_word(k, "default")) {
                    fail("expected case or default", k.pos);
                } else if (fallback) {
                    fail("a second default", k.pos);
                }
                expect(Tok::COLON, "':' after the case");
                NodePtr v = peek().text == "{" ? block("case") : expression(0);
                expect(Tok::SEMICOLON, "';' after the case");
                if (c) {
                    cases.emplace_back(std::move(c), std::move(v));
                } else {
                    fallback = std::move(v);
                }
            }
            if (cases.empty() && !fallback) fail("empty switch", peek().pos);
            next();
            NodePtr e = std::move(fallback);
            for (auto it = cases.rbegin(); it != cases.rend(); ++it) {
                e = conditional(std::move(it->first), std::move(it->second), std::move(e));
            }
            return e;
        }

        NodePtr symbol(const Token & t) {
            const auto v = std::find(variables.begin(), variables.end(), t.text);
            if (v != variables.end()) {
                auto n = std::make_unique<Node>();
                n->op = Op::VAR;
                n->index = static_cast<uint16_t>(v - variables.begin());
                return n;
            }
            if (const auto * l = local(t.text)) {
                auto n = node(Kind::LOCAL, {});
                n->index = l->second;
                if (peek().type == Tok::LPAREN) {
                    tokens.insert(tokens.begin() + static_cast<std::ptrdiff_t>(at), Token{Tok::OP, "*", peek().pos, true});
                }
                return n;
            }
            const std::string l = lower(t.text);
            if (l == "pi") return constant(3.14159265358979323846264338327950288419716939937510);
            if (l == "epsilon") return constant(0.000000000100);
            if (l == "inf") return constant(std::numeric_limits<double>::infinity());
            if (l == "true") return constant(1.0);
            if (l == "false") return constant(0.0);
            const auto f = functions().find(l);
            if (f == functions().end()) {
                std::string known;
                for (const auto & var : variables) {
                    if (!var.empty()) known += (known.empty() ? "" : ", ") + var;
                }
                fail("unknown symbol '" + t.text + "' (variables: " + known + ")" + expression_unit_hint(t.text), t.pos);
            }
            if (peek().type != Tok::LPAREN) fail("expected '(' after " + t.text, peek().pos);
            next();
            std::vector<NodePtr> args;
            if (peek().type != Tok::RPAREN) {
                args.push_back(expression(0));
                while (peek().type == Tok::COMMA) {
                    next();
                    args.push_back(expression(0));
                }
            }
            expect(Tok::RPAREN, "')' after the arguments of " + t.text);
            const Function & fn = f->second;
            const int n = static_cast<int>(args.size());
            if (n < fn.min_args || (fn.max_args >= 0 && n > fn.max_args)) {
                fail("wrong number of arguments of " + t.text, t.pos);
            }
            if (fn.max_args >= 0) return make(fn.op, std::move(args));
            // Variadic: folded left to right; avg is the sum over n
            const size_t count = args.size();
            NodePtr e = std::move(args[0]);
            const Op step = fn.op == Op::DIV ? Op::ADD : fn.op;
            for (size_t i = 1; i < count; i++) e = make(step, std::move(e), std::move(args[i]));
            if (fn.op == Op::DIV) e = make(Op::DIV, std::move(e), constant(static_cast<double>(count)));
            return e;
        }
};

/**
 * @brief Emits a syntax tree as a program whose local variables occupy the
 *        stack slots below base. Every node leaves one value on the stack; a
 *        loop keeps its value in a slot under its body's, where break and
 *        continue cut the stack back to.
 */
class Emitter {
    public:
        explicit Emitter(const uint16_t base_) : base(base_) {
            if (base > 0) code.push_back(Instr{0.0, base, Op::LOCALS});
        }

        void emit(const Node & n) {
            switch (n.kind) {
                case Kind::OP:
                    for (const auto & a : n.args) emit(*a);
                    put(Instr{n.value, n.index, n.op}, 1 - static_cast<int>(n.args.size()));
                    break;
                case Kind::LOCAL: put(Instr{0.0, n.index, Op::LOAD}, 1); break;
                case Kind::ASSIGN:
                    emit(*n.args[0]);
                    put(Instr{0.0, n.index, Op::STORE}, 0);
                    break;
                case Kind::SEQUENCE:
                    for (size_t i = 0; i < n.args.size(); i++) {
                        emit(*n.args[i]);
                        if (i + 1 < n.args.size()) put(Instr{0.0, 0, Op::POP}, -1);
                    }
                    break;
                case Kind::IF: {
                    emit(*n.args[0]);
                    const size_t to_else = put(Instr{0.0, 0, Op::JUMP_IF_NOT}, -1);
                    emit(*n.args[1]);
                    const size_t to_end = put(Instr{0.0, 0, Op::JUMP}, 0);
                    land(to_else);
                    depth--;
                    emit(*n.args[2]);
                    land(to_end);
                    break;
                }
                case Kind::WHILE:
                case Kind::FOR: {
                    const bool is_for = n.kind == Kind::FOR;
                    if (is_for) {
                        emit(*n.args[0]);
                        put(Instr{0.0, 0, Op::POP}, -1);
                    }
                    const uint16_t result = begin_loop();
                    const uint16_t start = here();
                    emit(*n.args[is_for ? 1 : 0]);
                    const size_t to_end = put(Instr{0.0, 0, Op::ITERATE}, -1);
                    emit(*n.args[is_for ? 3 : 1]);
                    put(Instr{0.0, result, Op::KEEP}, -1);
                    land_all(loops.back().continues);
                    if (is_for && n.args[2]) {
                        emit(*n.args[2]);
                        put(Instr{0.0, 0, Op::POP}, -1);
                    }
                    put(Instr{0.0, start, Op::JUMP}, 0);
                    land(to_end);
                    end_loop();
                    break;
                }
                case Kind::REPEAT: {
                    const uint16_t result = begin_loop();
                    const uint16_t start = here();
                    emit(*n.args[0]);
                    put(Instr{0.0, result, Op::KEEP}, -1);
                    land_all(loops.back().continues);
                    emit(*n.args[1]);
                    put(Instr{0.0, start, Op::UNTIL}, -1);
                    end_loop();
                    break;
                }
                case Kind::BREAK:
                    if (n.args.empty()) {
                        put(Instr{std::numeric_limits<double>::quiet_NaN(), 0, Op::CONST}, 1);
                    } else {
                        emit(*n.args[0]);
                    }
                    put(Instr{0.0, loops.back().result, Op::KEEP}, 0);
                    loops.back().breaks.push_back(put(Instr{0.0, 0, Op::JUMP}, 0));
                    break;
                case Kind::CONTINUE:
                    put(Instr{0.0, loops.back().result, Op::DROP}, 1);
                    loops.back().continues.push_back(put(Instr{0.0, 0, Op::JUMP}, 0));
                    break;
            }
        }

        std::vector<Instr> code;
        int max_depth = 0;  // of the operand stack, above the locals
        bool has_loops = false;

    private:
        struct Loop {
            uint16_t result;
            std::vector<size_t> breaks, continues;
        };
        uint16_t base;
        int depth = 0;
        std::vector<Loop> loops;

        size_t put(const Instr & in, const int delta) {
            code.push_back(in);
            depth += delta;
            max_depth = std::max(max_depth, depth);
            return code.size() - 1;
        }

        uint16_t here() const {
            if (code.size() > std::numeric_limits<uint16_t>::max()) throw std::length_error("expression too long");
            return static_cast<uint16_t>(code.size());
        }

        void land(const size_t jump) { code[jump].index = here(); }

        void land_all(const std::vector<size_t> & jumps) {
            for (const size_t j : jumps) land(j);
        }

        // The loop's value: 0 until its body first completes
        uint16_t begin_loop() {
            has_loops = true;
            put(Instr{0.0, 0, Op::CONST}, 1);
            loops.push_back(Loop{static_cast<uint16_t>(base + depth - 1), {}, {}});
            return loops.back().result;
        }

        void end_loop() {
            land_all(loops.back().breaks);
            loops.pop_back();
        }
};

} // namespace

Program::Program(const std::string & name, const std::string & text, const std::vector<std::string> & variables) :
    m_name(name), m_text(text), m_uses(variables.size(), false) {
    Parser parser(name, text, variables);
    const NodePtr root = parser.parse();
    Emitter emitter(parser.n_locals());
    try {
        emitter.emit(*root);
    } catch (const std::length_error &) {
        parser.fail("the expression is too long", 0);
    }
    m_code = std::move(emitter.code);
    m_loops = emitter.has_loops;
    const int slots = parser.n_locals() + emitter.max_depth;
    if (slots > MAX_STACK) {
        throw std::invalid_argument("Expression for " + name + " is too deeply nested or has too many variables (needs " +
                                    std::to_string(slots) + " stack slots, at most " + std::to_string(MAX_STACK) + ").");
    }
    for (const Instr & in : m_code) {
        if (in.op == Op::VAR) m_uses[in.index] = true;
    }
}

std::string Program::runaway_error() const {
    return "Expression for " + m_name + " (\"" + m_text + "\") stopped after " + std::to_string(MAX_ITERATIONS) +
           " loop iterations: check that its loops end.";
}

double Program::operator()(const double * vars) const {
    bool runaway = false;
    const double v = evaluate(m_code.data(), static_cast<uint32_t>(m_code.size()), ArrayVariables{vars}, runaway);
    if (runaway) throw std::invalid_argument(runaway_error());
    return v;
}

} // namespace expr
