/**
 * @file compiled_expression.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Parser of compiled expressions: exprtk's grammar and precedences
 *        (precedence climbing), constants folded, emitted as a stack program.
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
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>

#include <exprtk.hpp>

namespace expr {

namespace {

enum class Tok { NUMBER, NAME, OP, LPAREN, RPAREN, COMMA, QUESTION, COLON, END };

struct Token {
    Tok type;
    std::string text;
    size_t pos;
};

std::string lower(std::string s) {
    for (char & c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

bool is_name_start(const char c) { return std::isalpha(static_cast<unsigned char>(c)) || c == '_'; }
bool is_name_char(const char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; }

const std::vector<std::string> & keyword_operators() {
    static const std::vector<std::string> words = {"and", "or", "xor", "nand", "nor", "xnor"};
    return words;
}

bool is_keyword_operator(const std::string & name) {
    const std::string l = lower(name);
    const auto & w = keyword_operators();
    return std::find(w.begin(), w.end(), l) != w.end();
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
        {"if", {Op::SELECT, 3, 3}}, {"clamp", {Op::CLAMP, 3, 3}}, {"inrange", {Op::INRANGE, 3, 3}},
        {"min", {Op::MIN, 1, -1}}, {"max", {Op::MAX, 1, -1}}, {"sum", {Op::ADD, 1, -1}},
        {"mul", {Op::MUL, 1, -1}}, {"avg", {Op::DIV, 1, -1}},
    };
    return table;
}

struct Node {
    Op op = Op::CONST;
    double value = 0.0;
    uint16_t index = 0;
    std::vector<std::unique_ptr<Node>> args;
};
using NodePtr = std::unique_ptr<Node>;

NodePtr constant(const double v) {
    auto n = std::make_unique<Node>();
    n->value = v;
    return n;
}

bool all_constant(const Node & n) {
    for (const auto & a : n.args) {
        if (a->op != Op::CONST) return false;
    }
    return true;
}

double evaluate_constant(const Node & n) {
    std::vector<Instr> code;
    for (const auto & a : n.args) code.push_back(Instr{a->value, 0, Op::CONST});
    code.push_back(Instr{0.0, n.index, n.op});
    return evaluate(code.data(), static_cast<uint32_t>(code.size()), [](uint16_t) { return 0.0; });
}

/** @brief The node op(args), with exprtk's integer powers and constants folded. */
NodePtr make(const Op op, std::vector<NodePtr> args) {
    auto n = std::make_unique<Node>();
    n->op = op;
    n->args = std::move(args);
    if (op == Op::POW && n->args[1]->op == Op::CONST) {
        const double c = n->args[1]->value;
        if (std::abs(c) <= 60.0 && c == std::trunc(c)) {
            const uint16_t p = static_cast<uint16_t>(std::abs(c));
            if (p == 0) return constant(1.0);
            n->op = c >= 0.0 ? Op::IPOW : Op::IPOWINV;
            n->index = p;
            n->args.pop_back();
        }
    }
    if (op == Op::NEG && n->args[0]->op == Op::NEG) return std::move(n->args[0]->args[0]);
    if (all_constant(*n)) return constant(evaluate_constant(*n));
    return n;
}

NodePtr make(const Op op, NodePtr a) {
    std::vector<NodePtr> v;
    v.push_back(std::move(a));
    return make(op, std::move(v));
}

NodePtr make(const Op op, NodePtr a, NodePtr b) {
    std::vector<NodePtr> v;
    v.push_back(std::move(a));
    v.push_back(std::move(b));
    return make(op, std::move(v));
}

/** @brief exprtk's value of a numeric literal (its own conversion, not strtod). */
double literal_value(const std::string & text) {
    double v = 0.0;
    if (!exprtk::details::string_to_real(text, v)) throw std::invalid_argument("invalid number " + text);
    return v;
}

class Parser {
    public:
        Parser(const std::string & name, const std::string & text, const std::vector<std::string> & variables) :
            name(name), text(text), variables(variables) {
            tokenize();
        }

        NodePtr parse() {
            NodePtr e = expression(0);
            if (peek().type != Tok::END) fail("unexpected '" + peek().text + "'", peek().pos);
            return e;
        }

    private:
        const std::string & name;
        const std::string & text;
        const std::vector<std::string> & variables;
        std::vector<Token> tokens;
        size_t at = 0;

        [[noreturn]] void fail(const std::string & what, const size_t pos) const {
            throw std::invalid_argument("Failed to parse expression for " + name + " (\"" + text + "\"): " + what +
                                        " at position " + std::to_string(pos + 1) + ".");
        }

        const Token & peek() const { return tokens[at]; }
        Token next() { return tokens[at++]; }

        void expect(const Tok type, const std::string & what) {
            if (peek().type != type) fail("expected " + what, peek().pos);
            at++;
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
                } else if (c == ':' && !(i + 1 < text.size() && text[i + 1] == '=')) {
                    push(Tok::COLON, ":", start);
                    i++;
                } else {
                    static const std::vector<std::string> ops = {"<=", ">=", "==", "!=", "<>", "<", ">", "=",
                                                                 "+",  "-",  "*",  "/",  "%",  "^", "&", "|"};
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
                const bool name_like = type == Tok::NAME && !is_keyword_operator(t);
                const bool open = type == Tok::LPAREN;
                const bool implicit = (prev.type == Tok::NUMBER && (name_like || open)) ||
                                      (prev.type == Tok::RPAREN && (name_like || open || type == Tok::NUMBER)) ||
                                      (prev.type == Tok::NAME && open && !is_function(prev) && is_variable(prev));
                if (implicit) tokens.push_back(Token{Tok::OP, "*", pos});
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

        NodePtr expression(const int precedence) {
            NodePtr e = branch();
            for (;;) {
                int left = 0, right = 0;
                Op op = Op::CONST;
                if (!binary_operator(peek(), left, right, op) || left < precedence) break;
                next();
                e = make(op, std::move(e), expression(right));
            }
            if (precedence == 0 && peek().type == Tok::QUESTION) {
                next();
                NodePtr a = expression(0);
                expect(Tok::COLON, "':' of the ternary operator");
                NodePtr b = expression(0);
                std::vector<NodePtr> args;
                args.push_back(std::move(e));
                args.push_back(std::move(a));
                args.push_back(std::move(b));
                e = make(Op::SELECT, std::move(args));
            }
            return e;
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
                case Tok::NAME:
                    return symbol(t);
                default:
                    fail("unexpected " + t.text, t.pos);
            }
        }

        NodePtr symbol(const Token & t) {
            const auto v = std::find(variables.begin(), variables.end(), t.text);
            if (v != variables.end()) {
                auto n = std::make_unique<Node>();
                n->op = Op::VAR;
                n->index = static_cast<uint16_t>(v - variables.begin());
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
                for (const auto & var : variables) known += (known.empty() ? "" : ", ") + var;
                fail("unknown symbol '" + t.text + "' (variables: " + known + ")", t.pos);
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

void emit(const Node & n, std::vector<Instr> & code, int & depth, int & max_depth) {
    for (const auto & a : n.args) emit(*a, code, depth, max_depth);
    code.push_back(Instr{n.value, n.index, n.op});
    depth += 1 - static_cast<int>(n.args.size());
    max_depth = std::max(max_depth, depth);
}

} // namespace

Program::Program(const std::string & name, const std::string & text, const std::vector<std::string> & variables) :
    m_uses(variables.size(), false) {
    NodePtr root = Parser(name, text, variables).parse();
    int depth = 0, max_depth = 0;
    emit(*root, m_code, depth, max_depth);
    if (max_depth > MAX_STACK) {
        throw std::invalid_argument("Expression for " + name + " is too deeply nested (needs a stack of " +
                                    std::to_string(max_depth) + ", at most " + std::to_string(MAX_STACK) + ").");
    }
    for (const Instr & in : m_code) {
        if (in.op == Op::VAR) m_uses[in.index] = true;
    }
}

} // namespace expr
