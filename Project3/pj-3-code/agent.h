/**
 * Framework for NoGo and similar games (C++ 11)
 * agent.h: Define the behavior of variants of the player
 *
 * Author: Theory of Computer Games
 *         Computer Games and Intelligence (CGI) Lab, NYCU, Taiwan
 *         https://cgilab.nctu.edu.tw/
 */

#pragma once
#include <string>
#include <random>
#include <sstream>
#include <map>
#include <type_traits>
#include <algorithm>
#include <fstream>
#include "board.h"
#include "action.h"

class agent {
public:
	agent(const std::string& args = "") {
		std::stringstream ss("name=unknown role=unknown " + args);
		for (std::string pair; ss >> pair; ) {
			std::string key = pair.substr(0, pair.find('='));
			std::string value = pair.substr(pair.find('=') + 1);
			meta[key] = { value };
		}
	}
	virtual ~agent() {}
	virtual void open_episode(const std::string& flag = "") {}
	virtual void close_episode(const std::string& flag = "") {}
	virtual action take_action(const board& b) { return action(); }
	virtual bool check_for_win(const board& b) { return false; }

public:
	virtual std::string property(const std::string& key) const { return meta.at(key); }
	virtual void notify(const std::string& msg) { meta[msg.substr(0, msg.find('='))] = { msg.substr(msg.find('=') + 1) }; }
	virtual std::string name() const { return property("name"); }
	virtual std::string role() const { return property("role"); }

protected:
	typedef std::string key;
	struct value {
		std::string value;
		operator std::string() const { return value; }
		template<typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
		operator numeric() const { return numeric(std::stod(value)); }
	};
	std::map<key, value> meta;
};

/**
 * base agent for agents with randomness
 */
class random_agent : public agent {
public:
	random_agent(const std::string& args = "") : agent(args) {
		if (meta.find("seed") != meta.end())
			engine.seed(int(meta["seed"]));
	}
	virtual ~random_agent() {}

protected:
	std::default_random_engine engine;
};

/**
 * random player for both side
 * put a legal piece randomly
 */

class node {
public:
        board state;
        int win_count = 0;
        int visit_count = 0;
        double UCB_value = 0x7fffffff;
        node* parent = NULL;
        action::place last_action;
        std::vector<node*> children;
        board::piece_type who;

	~node(){};
};

class MCTS_player : public random_agent {
public:
	MCTS_player(const std::string& args = "") : random_agent("name=random role=unknown " + args),
		space(board::size_x * board::size_y),
	        white_space(board::size_x * board::size_y),
		black_space(board::size_x * board::size_y),
		who(board::empty) {
		if (name().find_first_of("[]():; ") != std::string::npos)
			throw std::invalid_argument("invalid name: " + name());
		if (meta.find("search") != meta.end()) agent_name = (std::string)meta["search"];
		if (meta.find("timeout") != meta.end()) timeout = (clock_t)meta["timeout"];
		if (meta.find("simulation") != meta.end()) simulation_count = (int)meta["simulation"];
		if (role() == "black") who = board::black;
		if (role() == "white") who = board::white;
		if (who == board::empty)
			throw std::invalid_argument("invalid role: " + role());
		for (size_t i = 0; i < space.size(); i++)
			space[i] = action::place(i, who);
		for (size_t i = 0; i < white_space.size(); ++i)
			white_space[i] = action::place(i, board::white);
		for (size_t i = 0; i < black_space.size(); ++i)
			black_space[i] = action::place(i, board::black);
	}

	void computeUCB(node* cur, int parent_visit_count) {						//count for UCT 
		double win_rate = (double) cur->win_count / (double) cur->visit_count;
		const float c = 0.5;
		double exploitation = sqrt(log((double)parent_visit_count)/cur->visit_count);
		cur->UCB_value = win_rate + c * exploitation;;
	}
	
	void expand(node* parent_node) {
		board::piece_type child_who;
		action::place child_move;
	 	
		if (parent_node->who == board::black) {
			child_who = board::white;
			for(const action::place& child_move : white_space) {
				board after = parent_node->state;
				if (child_move.apply(after) == board::legal) {
					node* child_node = new node;
					child_node->state = after;
					child_node->parent = parent_node;
					child_node->last_action = child_move;
					child_node->who = child_who;
					
					parent_node->children.emplace_back(child_node);
				}
			}
		}
		else if (parent_node->who == board::white) {
			child_who = board::black;
			for(const action::place& child_move : black_space) {
				board after = parent_node->state;
				if (child_move.apply(after) == board::legal) {
					node* child_node = new node;
					child_node->state = after;
					child_node->parent = parent_node;
					child_node->last_action = child_move;
					child_node->who = child_who;
					
					parent_node->children.emplace_back(child_node);
				}
			}
		}
			
	}
	
	node* selection(node* cur) {							//select which child node
		node* cur_node = cur;
		while(cur_node->children.empty() == false) {
			double max_UCB_value = 0;
			int select_idx = 0;
			for(size_t i = 0; i < cur_node->children.size(); ++i) {
				if(max_UCB_value < cur_node->children[i]->UCB_value) {
					max_UCB_value = cur_node->children[i]->UCB_value;
					select_idx = i;
				}
			}
			cur_node = cur_node->children[select_idx];
		}
		return cur_node;
	}
	
	//return which player win
	board::piece_type simulation(node* root) {
		bool terminal = false;
		board state = root->state;
		board::piece_type who = root->who;
		
		while(terminal == false) {
			terminal = true;
			
			who = (who == board::white ? board::black : board::white);
				
			if (who == board::black) {
				std::shuffle(black_space.begin(), black_space.end(), engine);
				for(const action::place& move : black_space) {
					board after = state;
					if (move.apply(after) == board::legal) {
						move.apply(state);
						terminal = false;
						break;
					}
				}
			}
			else if (who == board::white) {
				std::shuffle(white_space.begin(), white_space.end(), engine);
				for(const action::place& move : white_space) {
					board after = state;
					if (move.apply(after) == board::legal) {
						move.apply(state);
						terminal = false;
						break;
					}
				}
			}
		}
		return (who == board::white ? board::black : board::white);
	}
	
	void backpropagation(node* root, node* cur, board::piece_type winner) {
		// root state : last_action = white 
		// -> root who = black 
		bool win = true;
		if(winner == root->who)
			win = false;
		while(cur != NULL && cur != root) {
			++cur->visit_count;
			if(win == true)
				++cur->win_count;
			computeUCB(cur, cur->parent->visit_count+1);
			cur = cur->parent;
		}
		++root->visit_count;
		if(win == true)
			++root->win_count;
	}
	
	void MCTS(node* root, board::piece_type winner, int simulation_count){
		int cnt = 0;		
		while (cnt < simulation_count) {
			node* best_node = selection(root);

			expand(best_node);
			winner = simulation(best_node);

			backpropagation(root, best_node, winner);
			++cnt;
		}
	}

	action choose_Action(node* Root) {					//choose movement according to its visit count
		int child_idx = -1;
		int max_visit_count = 0;
		
		for(size_t i = 0; i < Root->children.size(); ++i) {
					
			if(Root->children[i]->visit_count > max_visit_count) {
				max_visit_count = Root->children[i]->visit_count;
				child_idx = i;
			}
		}
		
		if(child_idx == -1) return action();
		return Root->children[child_idx]->last_action;
	}
	
	void delete_tree(node* node) {
		if(node->children.empty() == false) {
			for(size_t i = 0; i < node->children.size(); ++i) {
				delete_tree(node->children[i]);
				if(node->children[i] != NULL)
					free(node->children[i]);
			}
			node->children.clear();
		}
		return;
	}


	virtual action take_action(const board& state) {
		if (agent_name == "random" or agent_name.empty()){
			std::shuffle(space.begin(), space.end(), engine);
			for (const action::place& move : space) {
				board after = state;
				if (move.apply(after) == board::legal)
					return move;
			}
			return action();
		}
		

		else if (agent_name == "MCTS"){
			clock_t start_time, end_time, total_time = 0;
			start_time = clock();
			
			node* root = new node;
			board::piece_type winner;
			
			root->state = state;
			root->who = (who == board::white ? board::black : board::white);
			expand(root);

			//std::cout << root->who << " is playing " << std::endl;
			
			if(simulation_count){
				MCTS(root,winner,simulation_count);				
			}
			else{
				while(total_time < timeout) {
				
					node* best_node = selection(root);
					expand(best_node);
					winner = simulation(best_node);
				
					backpropagation(root, best_node, winner);
					end_time = clock();
					total_time = (double)(end_time-start_time);
				}
			}

			

			action best_action = choose_Action(root);
			delete_tree(root);
			free(root);
			
			return best_action;
		}
		else {
			throw std::invalid_argument("assigned agent is not finished yet!!!");
		}
		
	}

private:
	std::vector<action::place> space, white_space, black_space;
	board::piece_type who;
	std::string agent_name;
	int simulation_count = 0;
	clock_t timeout = 1000;
	//int step_count = 0;
	//double time_schedule[36] = {0.2, 0.2, 0.2, 0.4, 0.4, 0.4,
	//			    0.7, 0.7, 0.7, 1.4, 1.4, 1.4,
	//			    1.7, 1.7, 1.7, 2.0, 2.0, 2.0,
	//		   	    1.7, 1.7, 1.7, 1.7, 1.7, 1.7,
	//			    1.0, 1.0, 1.0, 0.5, 0.5, 0.5,
	//			    0.4, 0.4, 0.4, 0.2, 0.2, 0.2};
};