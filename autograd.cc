#include <iostream>
#include <functional>
#include <random>
#include <iomanip>
#include <vector>

double gen_random_weight()
{
	static std::random_device rd;
	static std::mt19937 gen(rd());
	static std::uniform_real_distribution<double> dis(-1.0, 1.0);
	return dis(gen);
}

class Value : public std::enable_shared_from_this<Value>
{
public:
	Value(double data, std::vector<std::shared_ptr<Value>> prev = {}, std::string op = "")
	{
		this->m_data = data;
		this->m_grad = 0.0;
		this->m_prev = std::move(prev);
		this->m_op = op;
		this->_backward = [] {};
	}

	friend std::ostream &operator<<(std::ostream &ostr, Value &val)
	{
		ostr << std::setprecision(4) << std::fixed;
		ostr << "Value(data=" << val.get_data() << ", grad=" << val.get_grad() << ", op=\"" << val.get_op() << "\", label=\"" << val.get_label() << "\", prev=" << val.m_prev.size() << ")";

		return ostr;
	}

	std::shared_ptr<Value> pow(std::shared_ptr<Value> val)
	{
		auto out = std::make_shared<Value>(std::pow(m_data, val->get_data()), std::vector<std::shared_ptr<Value>>{shared_from_this(), val}, "**");
		out->_backward = [out, self = shared_from_this(), val]()
		{
			self->inc_grad(val->get_data() * std::pow(self->get_data(), (val->get_data() - 1)) * out->get_grad());
		};
		return out;
	}

	std::shared_ptr<Value> pow(double val)
	{
		return pow(std::make_shared<Value>(val));
	}

	void backward()
	{
		m_grad = 1.0;
		auto topo_order = build_topo();
		for (auto val : topo_order)
		{
			val->_backward();
		}
	}

	std::vector<std::shared_ptr<Value>> build_topo()
	{
		std::vector<std::shared_ptr<Value>> topo_order;
		build_topo(topo_order);
		clear_visit_mark(topo_order);
		return topo_order;
	}

	void clear_visit_mark(std::vector<std::shared_ptr<Value>> &topo_order)
	{
		for (auto &val : topo_order)
		{
			val->set_visited(false);
		}
	}

	void build_topo(std::shared_ptr<Value> val, std::vector<std::shared_ptr<Value>> &topo_order)
	{
		if (!val->get_visited())
		{
			val->set_visited(true);
			for (auto child : val->get_prev())
			{
				build_topo(child, topo_order);
			}
			topo_order.insert(topo_order.begin(), val);
		}
	}

	void build_topo(std::vector<std::shared_ptr<Value>> &topo_order)
	{
		build_topo(shared_from_this(), topo_order);
	}

	std::vector<std::shared_ptr<Value>> get_prev()
	{
		return m_prev;
	}

	double get_data()
	{
		return m_data;
	}

	std::string get_op()
	{
		return m_op;
	}

	void set_label(std::string label)
	{
		m_label = label;
	}

	std::string get_label()
	{
		return m_label;
	}

	void set_grad(double grad)
	{
		m_grad = grad;
	}

	void inc_grad(double grad)
	{
		m_grad += grad;
	}

	double get_grad()
	{
		return m_grad;
	}

	bool get_visited()
	{
		return m_visited;
	}

	void set_visited(bool val)
	{
		m_visited = val;
	}

	std::function<void()> _backward;

private:
	double m_data;
	double m_grad;
	bool m_visited;
	std::vector<std::shared_ptr<Value>> m_prev;
	std::string m_op;
	std::string m_label;
};

std::shared_ptr<Value> operator+(std::shared_ptr<Value> lhs, std::shared_ptr<Value> rhs)
{
	auto prev = std::vector<std::shared_ptr<Value>>{lhs, rhs};
	auto out = std::make_shared<Value>(lhs->get_data() + rhs->get_data(), prev, "+");
	out->_backward = [out, lhs, rhs]()
	{
		// std::cout << "lhs :: " << *lhs << "\n";
		// std::cout << "rhs :: " << *rhs << "\n";
		lhs->inc_grad(1.0 * out->get_grad());
		rhs->inc_grad(1.0 * out->get_grad());
	};
	return out;
}

std::shared_ptr<Value> operator*(std::shared_ptr<Value> lhs, std::shared_ptr<Value> rhs)
{
	auto prev = std::vector<std::shared_ptr<Value>>{lhs, rhs};
	auto out = std::make_shared<Value>(lhs->get_data() * rhs->get_data(), prev, "*");
	out->_backward = [out, lhs, rhs]()
	{
		lhs->inc_grad(rhs->get_data() * out->get_grad());
		rhs->inc_grad(lhs->get_data() * out->get_grad());
	};
	return out;
}

// multi-directional overloading for ops
// add op
std::shared_ptr<Value> operator+(std::shared_ptr<Value> lhs, double rhs)
{
	return lhs + std::make_shared<Value>(rhs);
}
std::shared_ptr<Value> operator+(double lhs, std::shared_ptr<Value> rhs)
{
	return rhs + lhs;
}

// mul op
std::shared_ptr<Value> operator*(std::shared_ptr<Value> lhs, double rhs)
{
	return lhs * std::make_shared<Value>(rhs);
}
std::shared_ptr<Value> operator*(double lhs, std::shared_ptr<Value> rhs)
{
	return rhs * lhs;
}

// div op
std::shared_ptr<Value> operator/(std::shared_ptr<Value> lhs, std::shared_ptr<Value> rhs)
{
	return lhs * rhs->pow(-1);
}
std::shared_ptr<Value> operator/(std::shared_ptr<Value> lhs, double rhs)
{
	return lhs / std::make_shared<Value>(rhs);
}
std::shared_ptr<Value> operator/(double lhs, std::shared_ptr<Value> rhs)
{
	return std::make_shared<Value>(lhs) / rhs;
}

// negate
std::shared_ptr<Value> operator-(std::shared_ptr<Value> rhs)
{
	return rhs * std::make_shared<Value>(-1.0);
}

std::shared_ptr<Value> operator-(std::shared_ptr<Value> lhs, std::shared_ptr<Value> rhs)
{
	return lhs + (-rhs);
}

std::shared_ptr<Value> tanh(std::shared_ptr<Value> lhs)
{
	// TODO: FINISH WORKING ON THIS
	auto t = (exp(2 * lhs->get_data()) - 1) / (exp(2 * lhs->get_data()) + 1);
	auto prev = std::vector<std::shared_ptr<Value>>{lhs};
	auto out = std::make_shared<Value>(t, prev, "tanh");
	out->_backward = [out, lhs, t]()
	{
		lhs->inc_grad(1 - pow(t, 2) * out->get_grad());
	};
	return out;
}

// NEURON IMPLEMENTATION
class Neuron
{
public:
	Neuron(int nin)
	{
		this->m_weights.reserve(nin);
		for (size_t i = 0; (int)i < nin; i++)
		{
			auto weight = std::make_shared<Value>(gen_random_weight());
			this->m_weights.emplace_back(weight);
		}
	}

	std::shared_ptr<Value> operator()(std::vector<std::shared_ptr<Value>> &x)
	{
		std::shared_ptr<Value> acc = std::make_shared<Value>(0.0);
		for (size_t i = 0; i < x.size(); i++)
		{
			acc = acc + (x[i] * this->m_weights[i]);
		}
		acc = acc + this->m_bias;
		auto out = tanh(acc);
		return out;
	}

	void print_params()
	{
		std::cout << "weights: ";
		for (auto &weight : this->m_weights)
		{
			std::cout << weight->get_data() << ",";
		}
		std::cout << "\nbias: " << this->m_bias->get_data() << "\n";
	}

	std::vector<std::shared_ptr<Value>> parameters()
	{
		std::vector<std::shared_ptr<Value>> params;
		params.reserve(this->m_weights.size() + 1);

		for (size_t i = 0; i < this->m_weights.size(); i++)
		{
			params.emplace_back(this->m_weights[i]);
		}
		params.emplace_back(this->m_bias);
		return params;
	}

private:
	std::vector<std::shared_ptr<Value>> m_weights;
	std::shared_ptr<Value> m_bias = std::make_shared<Value>(gen_random_weight());
};

auto main() -> int
{
	// auto x1 = std::make_shared<Value>(2.0);
	// x1->set_label("x1");
	// auto x2 = std::make_shared<Value>(0.0);
	// x2->set_label("x1");
	// auto w1 = std::make_shared<Value>(-3.0);
	// w1->set_label("w1");
	// auto w2 = std::make_shared<Value>(1.0);
	// w2->set_label("w2");

	// auto b = std::make_shared<Value>(6.881373587019543);
	// b->set_label("b");

	// auto x1w1 = (x1 * w1);
	// auto x2w2 = (x2 * w2);
	// auto x1w1x2w2 = x1w1 + x2w2;
	// auto n = x1w1x2w2 + b;
	// auto o = tanh(n);
	// o->backward();

	// std::cout << "o :: " << *o << "\n";
	// std::cout << "n :: " << *n << "\n";
	// std::cout << "b :: " << *b << "\n";
	// std::cout << "x1w1x2w2 :: " << *x1w1x2w2 << "\n";
	// std::cout << "x1w1 :: " << *x1w1 << "\n";
	// std::cout << "x2w2 :: " << *x2w2 << "\n";
	// std::cout << "x1 :: " << *x1 << "\n";
	// std::cout << "w1 :: " << *w1 << "\n";
	// std::cout << "x2 :: " << *x2 << "\n";
	// std::cout << "w2 :: " << *w2 << "\n";

	auto x1 = std::make_shared<Value>(1.0);
	auto x2 = std::make_shared<Value>(2.0);
	auto vec = std::vector<std::shared_ptr<Value>>{x1, x2};
	auto n = Neuron(2);
	n.print_params();
	std::cout << *n(vec) << "\n";

	return 0;
}
