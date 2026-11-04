# LLM-Based Agent Society Investigation: Collaboration and Confrontation in Avalon Gameplay

Yihuai Lan<sup>1</sup>∗, Zhiqiang Hu<sup>3</sup>∗, Lei Wang<sup>4</sup>, Yang Wang<sup>5</sup>, Deheng Ye<sup>6</sup>, Peilin Zhao<sup>6</sup>, Ee-Peng Lim<sup>4</sup>, Hui Xiong<sup>1,2</sup>, Hao Wang<sup>1</sup>†

<sup>1</sup>The Hong Kong University of Science and Technology (Guangzhou)

<sup>2</sup>The Hong Kong University of Science and Technology

<sup>3</sup>Singapore University of Technology and Design

<sup>4</sup>Singapore Management University, <sup>5</sup>Verily Life Sciences, <sup>6</sup>Tencent {yihuailan, haowang}@hkust-gz.edu.cn

## Abstract

This paper explores the open research problem of understanding the social behaviors of LLM-based agents. Using Avalon as a testbed, we employ system prompts to guide LLM agents in gameplay. While previous studies have touched on gameplay with LLM agents, research on their social behaviors is lacking. We propose a novel framework, tailored for Avalon, features a multi-agent system facilitating efficient communication and interaction. We evaluate its performance based on game success and analyze LLM agents’ social behaviors. Results affirm the framework’s effectiveness in creating adaptive agents and suggest LLM-based agents’ potential in navigating dynamic social interactions. By examining collaboration and confrontation behaviors, we offer insights into this field’s research and applications. Our code is publicly available at https://github.com/ 3DAgentWorld/LLM-Game-Agent.

## 1 Introduction

Artificial intelligence (AI) agents (Xi et al., 2023; Park et al., 2023) exhibit human-like behaviors, from perceiving and analyzing the environment to decision-making and action-taking.

Advances in large language models (LLMs) (Kasneci et al., 2023; Peng et al., 2023; Touvron et al., 2023; Vaswani et al., 2017) offer new avenues for creating AI agents in complex environments, potentially simulating human society. Various works (Gao et al., 2023; Qian et al., 2023; Park et al., 2023; Ghaffarzadegan et al., 2023) simulate different aspects of human society. For instance, Qian et al. (Qian et al., 2023) simulate a software development company with agents representing diverse social identities. Park et al. (Park et al., 2023) assign varied social roles to agents within a sandbox environment. However, prior studies mostly examine positive social behaviors like honesty and collaboration, leaving research on negative social behaviors of LLM agents relatively scarce.

Previous research on human society has highlighted issues like misinformation and online conflicts, leading to efforts to address these problems (Song and Jiang, 2022; Levy et al., 2022; Chen et al., 2022). To delve deeper into the social behaviors of LLM agents, we intend to comprehensively investigate both positive and negative aspects of their conduct. To achieve this, we employ Avalon as the environment to illustrate collaboration and confrontation among agents. Avalon, a representative social deduction game, assigns players hidden roles and divides them into opposing teams. Throughout gameplay, players partake in discussions, debates, and strategic maneuvers.

LLM agents face a challenging task in winning the incomplete information game of Avalon. They need to share and obtain information via communication and analysis, deducing other players’ roles, building trust among allies, and deceiving opponents. Success requires technical abilities like natural language understanding, incomplete information analysis, and strategy learning. Additionally, social behaviors such as teamwork, persuasion, and camouflage are crucial for success in Avalon gameplay.

To investigate the LLM-based agent society, we propose a novel framework for the agents to play Avalon. Specifically, we adopt ChatGPT as the players and assign various roles to agents. We adopt system prompts to guide LLM agents to play Avalon automatically.

Following human’s thinking methodology, we incorporate multiple modules, including memory storage and summarization, analysis and planning, game action and response generation, and experience learning. We utilize a competitive baseline approach (Xu et al., 2023a), to elaborate the efficacy of our proposed framework. We also carefully analyze the social behaviors of LLM agents, and observe clear collaboration and confrontation between agents during the gameplay.

<table><tr><td>Method</td><td></td><td>Memory Analysis Plan Action Experience</td><td></td><td>Learning</td><td></td><td>Leadership Persuasion Camouflage Teamwork Confrontation Sharing</td><td></td><td></td><td></td></tr><tr><td>GenAgents (Park et al., 2023)</td><td>√</td><td></td><td>√</td><td>√</td><td>√</td><td></td><td>√</td><td></td><td>√</td></tr><tr><td>Plan4MC (Yuan et al., 2023)</td><td></td><td></td><td>√</td><td>√</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GITM (Zhu et al., 2023)</td><td>√</td><td></td><td>√</td><td>√</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>RGAgent (Akata et al., 2023)</td><td>√</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CGAgent (Xu et al., 2023a)</td><td>√</td><td>√</td><td></td><td></td><td>√</td><td></td><td></td><td></td><td></td></tr><tr><td>ReCon (Wang et al., 2023c)</td><td>V</td><td>√</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LARL (Xu et al., 2023b)</td><td>√</td><td>√</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CodeAct (Shi et al., 2023)</td><td>√</td><td>√</td><td></td><td>√</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Ours</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td></td><td>√</td><td>√</td></tr></table>

Table 1: Comparison between our work and related works in both agent framework and social behaviour analysis.

Our contributions can be summarized as:

• We explore the social behaviors exhibited by LLM-based agents in the context of Avalon gameplay. We reveal the various aspects of these behaviors, including teamwork, leadership, persuasion, camouflage, and confrontation.

• We design an effective framework to play Avalon, which presents superior performance compared with the baseline method. We also carefully analyse the relationship between the module design and agents’ social behaviors, providing comprehensive experiment discussions.

• Our findings have the potential to contribute to a better understanding of the role of LLMbased agents in social and strategic contexts, and shed light on the implications of these behaviors in such environments.

## 2 Related Work

## 2.1 Social Deduction Game Agent

The emergence of communication among agents in social deduction games (SDG) has garnered significant attention in the research community. Hirata et al. (2016) introduces an AI-based agent for the Werewolf game, aiming to advance intelligence and communication skills in AI systems. Nakamura et al. (2016) proposes a psychological model considering multiple perspectives to simulate human gameplay in The Werewolf. Wang and Kaneko (2018) addresses decision-making challenges in the Werewolf game using deep reinforcement learning techniques. Furthermore, Wiseman and Lewis (2019) explores player decision-making in social deduction games, focusing on sources of information influencing player strategies. Examining the broader context of multi-agent communication,

Liang et al. (2020) investigates the impact of competition on communication protocols. Brandizzi et al. (2021) explores the utilization of communication to foster cooperation in SDGs.

## 2.2 LLM-Based Gameplay

The rapid development of LLM-based agents has resulted in significant advancements in problemsolving across various domains. These agents, known for their quick and strategic processing, have improved the effectiveness and robustness of solving tasks (Lin et al., 2023; Wang et al., 2023b; Tsai et al., 2023; Zhou et al., 2023; Park et al., 2023; Qian et al., 2023; Fu et al., 2023).

LLMs have recently been utilized in various gaming environments, including task-based games like Minecraft and multiplayer strategy games (Yuan et al., 2023; Zhu et al., 2023; Wang et al., 2023a; Akata et al., 2023; Xu et al., 2023a; Wang et al., 2023c). In multiplayer strategy games such as the Prisoner’s Dilemma and Battle of the Sexes, LLMs model strategic interactions (Akata et al., 2023). They’re also employed in social deduction games like Werewolf and Avalon (Xu et al., 2023a; Wang et al., 2023c; Shi et al., 2023; Xu et al., 2023b), where they exhibit strategic behaviors. To combat misinformation, recursive contemplation has been proposed (Wang et al., 2023c). However, previous works have only partially analyzed behaviors and designed agent frameworks based on limited game characteristics. Thus, we propose a comprehensive social deduction game agent framework based on LLMs and conduct a thorough behavior analysis. Table 1 illustrates the distinctions between our work and others.

## 2.3 LLMs’ Impact on Society

The growing influence of Large Language Models (LLMs) on society has spurred significant research (Movva et al., 2023). Innovations include using LLMs for virtual social network simulations to advance social science research (Gao et al., 2023) and enrich human social experiences in virtual spaces (Kaiya et al., 2023). However, concerns arise regarding validity, privacy, and ethics in LLMdriven social computing. Ghaffarzadegan et al. propose feedback mechanisms to address these concerns (Ghaffarzadegan et al., 2023). Additionally, LLMs fuel advancements in social robot development (Yang and Menczer, 2023), posing challenges like social bot detection and misinformation spread. Ongoing research aims to align LLMs with ethical standards, mitigate biases and errors, and ensure their reliable and ethical use across diverse applications (Wang et al., 2023d; Liu et al., 2023).

## 3 Background

In our study, we chose Avalon, also known as “The Resistance”, instead of Werewolf as our environment. Unlike Werewolf, where players are gradually eliminated, Avalon ensures that all players remain engaged throughout the game, promoting social cohesion.

Avalon accommodates 5 to 10 players, focusing on the 6-player variant herein. Players receive secret roles in either the good or evil faction. The good faction includes Merlin, Percival, and Loyal Servants, while the evil faction comprises Morgana and Assassin. Morgana and Assassin know each other’s identities, Percival can identify Merlin and Morgana, and Merlin recognizes all evil players. The game spans 3-5 rounds. Players discuss and vote to form a quest team of 2-3 members. Approval requires a majority vote; otherwise, leadership shifts. Each round allows up to five voting cycles before the leader selects the team. Quest success hinges on cards submitted by team members. Good players submit success cards, while evil players can choose success or failure cards. A quest fails if it receives a failure card. The game concludes with victory for good players if three quests succeed, or for evil players if three quests fail. Evil players can also win by correctly identifying Merlin at the game’s end.

## 3.1 Social Behaviors in Avalon

Teamwork. Good players must collaborate to complete quests for winning. They should build trust with teammates while being wary of evil players. Leadership. Each player has the chance to lead the discussion for forming the quest team. The leader can guide the conversation and build trust among players. Effective leadership is crucial for victory. Persuasion. Players must use their communication skills to persuade others to believe their claims, trust their judgments, and support their decisions. Camouflage. Evil players pretend to be good players, using deceptive tactics and concealing information to mislead others.

Confrontation. Disagreements and conflicts will arise during the game. Players must tackle these confrontations and work towards resolving them. Sharing. Each role has unique clues. Sharing these clues promotes collaboration and builds trust among players, but risks exposing one’s identity.

## 4 Approach

## 4.1 Setup

Figure 1 shows the proposed framework. All prompts used are shown in Appendix Table 4. To start the game, system prompts are used to assign different roles to LLM agents. Each system prompt for a role $p _ { i }$ includes several important components: Role Information $\mathcal { R } \mathcal { L } ^ { p _ { i } }$ (Role Name and Role Introduction), Goal $\mathcal G ^ { p _ { i } }$ (Winning Conditions), and Abstracted Strategy ${ \mathcal { S } } ^ { p _ { i } }$ for gameplay. The Role Name and Role Introduction provide information about the assigned role to the LLM agent, while the Goal (Winning Conditions) offers insights into how to achieve victory. Additionally, the Initial Playing Strategy outlines the high-level planning for the LLM agent to take specific actions during gameplay.

Below is a specific example of a system prompt for the role of Margana:

Role: Morgana.

Role Introduction: In identification phase, you can identify teammates and the Assassin. Goal: Win the game by intentionally causing quests tofailfor three rounds, alone or with teammates. Initial Strategy: You always pretend to be a loyal servant and recommend yourselfas a candidatefor quests, and let the quests fail.

## 4.2 Memory Storage

Analyzing game history is vital for agents to grasp the current situation and make decisions. Yet, in Avalon, LLM agents’ history responses are often too lengthy, surpassing input limits and potentially lowering performance. To tackle this, a memory storage system is introduced to record conversations among LLM agents, enabling subsequent analysis and decision-making.

Memory Storage. Memory storage is vital for recording agents’ conversation history in the current game round. It comprises structured memory objects containing key details like role name, detailed natural language responses, round number, and a flag indicating public or private status. Public information is visible to all roles, while private information pertains to each role’s conversation. We assign separate memory pools to each agent for clarity in information processing. By storing this data, memory storage enables agents to access and review past conversations, improving their understanding of the game’s progress.

![](images/1c1607c0cc8c524ba2843809efde3f770c1b8ccb1e707a9628943f8ecd2bcffc.jpg)  
Figure 1: Our framework has six modules: summary, analysis, planning, action, response, and experiential learning. This design follows human thinking, helps LLM agents play Avalon effectively, and reveals their social behaviors.

## 4.3 Memory Summarization.

To store more information in memory, we use a summarization prompt to compress the information from the previous round and capture the essential details. The process of updating the memory with a summary of the previous round is illustrated below:

$$
\mathcal { M } _ { t } = \langle \mathrm { S M R } ( \mathcal { M } _ { t - 1 } ) , ( \mathcal { R } _ { t } ^ { p _ { 1 } } \cdot \cdot \cdot , \mathcal { R } _ { t } ^ { p _ { 6 } } , \mathbb { Z } _ { t } ) \rangle .\tag{1}
$$

The memory on round t is $\mathcal { M } _ { t }$ . The response generated by the LLM for role $p _ { i }$ on round t is $\mathcal { R } _ { t } ^ { p _ { i } }$ and $\mathcal { L } _ { t }$ represents the instructions and statements of the host on round t. is Text concatenation. SMR( ) is the summarization prompting.

## 4.4 Analysis

To help LLM agents improve strategic planning and increase their chances of winning, we introduce an

analysis module. This module analyzes the role identity and potential strategies of other players during gameplay:

$$
\mathcal { H } _ { t } ^ { p _ { i } } = \mathrm { A N A } \left( \mathcal { M } _ { t } , \mathcal { R } \mathcal { T } ^ { p _ { i } } \right) ,\tag{2}
$$

where $\mathcal { M } _ { t }$ is the memory on round t and $\mathcal { R } \mathcal { L } ^ { p _ { i } }$ is the role information. By analyzing, LLM agents can better understand their collaborators and competitors, leading to improved decision-making and effective counterstrategies for winning.

## 4.5 Planning

Agents need to understand the game progress and necessary strategies to win. Thus, a planning module is designed to create a strategic plan. The plan is based on the memory and information from the current round of the game, as described below:

$$
\mathcal { P } _ { t } ^ { p _ { i } } = \mathrm { P L A N } \left( \mathcal { M } _ { t } , \mathcal { H } _ { t } ^ { p _ { i } } , \mathcal { P } _ { t - 1 } ^ { p _ { i } } , \mathcal { R } \mathcal { Z } ^ { p _ { i } } , \mathcal { G } ^ { p _ { i } } , \mathcal { S } ^ { p _ { i } } \right) .\tag{3}
$$

where $\mathcal { P } _ { t } ^ { p _ { i } }$ represents the strategic plan of agent $p _ { i }$ at round t. ${ \mathcal { G } } ^ { p _ { i } }$ and ${ \mathcal { S } } ^ { p _ { i } }$ are goals and initial strategies. By creating a strategic plan, the agents can have a flexible strategy for different situations. This foresight helps them make better decisions about collaborating with teammates, deceiving opponents, taking on the opposing faction’s identity, and, if needed, sacrificing teammates or oneself to secure winning in the game.

## 4.6 Action

In the action module, agents decide their next action based on memory information, situation analysis, and the strategic plan. There are five types of actions: selecting players, voting (agree or disagree), completing quests (succeed or fail), using non-verbal signals (raising hands, putting hands down, opening or closing eyes), and choosing to remain silent. The process of choosing the next action is as follows:

$$
\mathcal { A } _ { t } ^ { p _ { i } } \sim p \left( \boldsymbol { A } | \mathcal { M } _ { t } , \mathcal { H } _ { t } ^ { p _ { i } } , \mathcal { P } _ { t } ^ { p _ { i } } , \mathcal { R } \mathcal { T } ^ { p _ { i } } , \mathcal { G } ^ { p _ { i } } , \mathcal { S } ^ { p _ { i } } , \mathcal { T } _ { t } ^ { \prime } \right) .\tag{4}
$$

The subsequent action depends on the memory, the comprehensive analysis, the strategic plan, and the instruction from the host. The details of these action decisions are confidential and only known to the respective agent. The host and other players cannot see these decisions.

## 4.7 Response Generation

The Response Generation module is responsible for generating a response to the host’s inquiry. Agents in this module choose an action and provide an explanation to the host. Agents are given the freedom to collaborate, deceive, and assume the identity of the opposite faction in their explanations.

## 4.8 Experience Learning

In practical scenarios, players can improve their Avalon gameplay strategy through experience. They gain insights not only from their own perspective but also by observing other players’ strategies. An ideal Avalon LLM agent should learn from both its own experiences and those of other players.

## 4.8.1 Self-Role Strategy Learning

In Step 1, agents generate three strategic recommendations for a player’s role-specific gameplay in Avalon games based on the game history. Agents avoid mentioning specific players and instead use role names to make the suggestions applicable in future games. In Step 2, agents enhance their strategies by incorporating the gathered suggestions while maintaining the original strategy’s strengths.

## 4.8.2 Other-Role Strategy Learning

Avalon LLM agents summarize the strategies adopted by other players to facilitate learning from the strategies employed by other players. Prompts for the above steps are shown in Appendix Table 5.

## 5 Experiment

## 5.1 Implementation Details

We developed the Avalon game program in Python, using the gpt-3.5-turbo-16k model as both our backend and the baseline’s. In all experiments, we set the agent model’s temperature to 0.3 and the LLM extractor’s to 0. The number of suggestions generated for updating strategies is 3. Game rules and role descriptions were set according to the baseline template (Xu et al., 2023a), which leverages historical context, enhances agent reasoning, and learns from past mistakes. Detailed descriptions are provided in Section A.2.

## 5.2 Evaluation Metrics

We evaluate the performance of our framework based on metrics from two perspectives.

## 5.2.1 Gameplay Outcome and Strategy.

From this perspective, we use metrics associated with the gameplay outcome and strategies to quantitatively evaluate the performance of the proposed agents and the baseline agents.

Winning Rate (WR). The winning rate is the percentage of games won out of the total played, calculated by dividing the number of wins by the total games played:

$$
W R = ( \frac { \# W i n s } { \# G a m e s ~ P l a y e d } ) \times 1 0 0 \%\tag{5}
$$

Quest Engagement Rate (QER). "Quest engagement rate" is the ratio of rounds a player joins the quest team to the total rounds played in the games. It’s calculated as follows:

$$
Q E R = ( \frac { \# E n g a g e m e n t R o u n d s } { \# R o u n d s } ) \times 1 0 0 \%\tag{6}
$$

Failure Vote Rate (FVR) The quest result relies on success or failure cards from team members. The failure vote rate indicates the percentage of votes against quest success, calculated as follows:

$$
F V R = ( \frac { \# F a i l u r e V o t e s } { \# V o t e s } ) \times 1 0 0 \%\tag{7}
$$

## 5.2.2 Social Behaviors.

From this perspective, we use ChatGPT to assist the analysis on the social behaviors of agents.

Leadership. We gauge AI agents’ leadership using "Leader Approval Rate (LAR)". LAR is calculated by dividing total approval votes by total leader votes across 20 Avalon games. It reflects consensus among players on proposed quest teams. Persuasion. To evaluate LLM agents’ persuasion, we track two metrics: self-recommendation rate (proposing oneself for quests) and success rate (self-recommendation for quest participation).

<table><tr><td>Method</td><td>Good Side</td><td>Evil Side</td></tr><tr><td>Ours</td><td>90</td><td>100</td></tr><tr><td>w/o analysis</td><td>60</td><td>60</td></tr><tr><td>w/o plan</td><td>80</td><td>100</td></tr><tr><td>w/o action</td><td>100</td><td>80</td></tr><tr><td>w/o strategy learning</td><td>50</td><td>60</td></tr></table>

Table 2: Results of the gameplay between ours and baseline. We present the winning rates (WR) of our method being good and evil sides.  
![](images/8ef4140cbd5a81d583d0641ee95b786231ccb5f89b72ff1d697f2288f35403d6.jpg)

![](images/439de46d8079f04b551403ccb47a9be34af395ecf564c9b5f324fdfc3762f3db.jpg)  
Figure 2: (a): Comparison of the engaging quests rate when playing evil side. Higher engaging quests rate means more opportunities for the player to influence the outcome of the game. (b): Comparison of the failure vote rate when playing evil side. Baseline is worse.

Camouflage. Detecting camouflage in AI agents is challenging. We focus on identifying instances where agents assume different identities in the initial round of each game. Behaviors include Self-Disclosure, Camouflage, and Withholding Identity. Teamwork and Confrontation.We use ChatGPT to analyze role responses, aiming to identify instances of collaboration or confrontation. Chat-GPT prompts with a player’s response and evaluates trust (teamwork), lack of trust (confrontation), or ambivalence towards others.

Sharing. Sharing reflects how often agents disclose valuable information, crucial for team cooperation. Using ChatGPT, we analyze agents’ dialogues to identify instances of sharing behavior, aiming to quantify their willingness to share for the team’s benefit.

## 5.3 Experiment Results

To validate the efficacy of Avalon AI agents, we repurposed Werewolf AI agents (Xu et al., 2023a) as baselines. Across two sets of 10 consecutive Avalon games, our agents faced off against the baselines, with Evil versus Good and vice versa.

After the matches, we compared the winning rates of our Avalon AI agents to the baselines. As depicted in Table 2, our method demonstrated a 90% winning rate in 10 games when playing the good side. Conversely, when playing the evil side, the winning rate was 100% over the same number of games.

Ablation studies reveal the importance of key modules in our AI agents. Removing the analysis module lowered winning rates to 60% for both sides, showing its impact on understanding and decision-making. Excluding the planning module reduced the good side’s winning rate to 80%, highlighting its role in devising strategies. Without the action module, the good side won 100% while the evil side dropped to 80%, indicating its importance for the evil side’s success. Removal of the strategy learning module led to winning rates decreasing to 50% and 60% for good and evil respectively, emphasizing its role in enhancing strategies. In conclusion, the analysis and strategy learning modules significantly influence game outcomes, affecting both sides’ winning rates. Additionally, the planning and action modules are crucial for success, given their impact on gameplay.

To better grasp the strategies employed by our Avalon Agents and the baseline agent, we compared quest engagement and failure voting rates when different AI agents acted as the evil side. Both rates significantly impact game outcomes. A higher quest engagement rate allows more chances for players to influence the game, while a higher failure voting rate suggests a greater chance for the evil side to win but also increases the risk of exposure, indicating an aggressive gameplay approach. Figure 2 illustrates the outcomes for quest engagement and failure voting rates. Our AI agents, particularly when playing as Morgana and Assassin, show assertiveness, with a 40.3% quest engagement rate and 84.0% failure voting rate. In comparison, baseline agents have lower rates at 33.1% and 36.5% respectively. As a result, our proposed Avalon AI agents achieve a 100% win rate against the baseline agents when playing as the evil side.

## 6 Social Behaviors of AI Agents

To evaluate if AI agents replicate human social behaviors in Avalon, we conduct a thorough analysis. This involves assessing the agents’ execution of teamwork, leadership, persuasion, camouflage, and confrontation through the frequency distribution in game logs from two sets of 10 consecutive games.

![](images/1ea2c22f4a0f3ac0f3a80cea1186d7775fee9ca4c26b1725e80339398f90ff05.jpg)

Figure 3: (a): The leadership behavior. Players with higher Leader Approval Rate get more agreements from other players when deciding a quest team. (b) and (c): The persuasion behavior. Self-recommendation Rate: players with higher Self-recommendation Rate are more will to engage in quests. Self-recommendation Success Rate: players more likely to gain the trust of other players has higher Self-recommendation Success Rate.  
![](images/a669c01686da4da30d5d0a98cf3935c143e576b02514b33937763b0ad014023c.jpg)  
Figure 4: The camouflage behavior when playing different roles: at first round of each game, the distribution of the players choose Self-Disclosure, Camouflage or Withholding Identity.

## 6.1 Leadership

Leadership skills come into play when players take charge of discussions and decision-making processes. A good leader can steer the conversation, guide suspicions, and rally the loyal servants to make informed decisions. Leadership abilities are crucial for the good side to effectively counter the deceptive tactics employed by the evil side.

Figure 3 (a) illustrates the Leader Approval Rate when agents assume various roles. It is evident that our agents, playing on the good side, attain remarkably high Leader Approval Rates when serving as leaders. Notably, the AI agents achieve a Leader Approval Rate exceeding 80% averagely while undertaking roles associated with the good side. This signifies their robust leadership qualities and their proactive approach to steering the gameplay towards victory. However, the baseline agents could propose good side players to the quest team to achieve high Leader Approval Rate but low game win rate.

## 6.2 Persuasion

Figure 3 displays the evaluation outcomes assessing the AI agents’ persuasion ability. Notably, agents employ distinct strategies based on their assumed roles, as shown in Figure 3 (b). When playing as Loyal Servant and Morgana, agents display a high self-recommendation rate for quest team participation, impacting mission success. Conversely, a cautious approach is seen with roles like Merlin, Percival, and Assassin, evident from their low self-recommendation rates. This strategic restraint is crucial, particularly for roles like Merlin, emphasizing the importance of concealing identity. From Figure 3 (c), Loyal Servants exhibit higher success rates in self-recommendation compared to roles that easily raise suspicion. Additionally, the proposed Avalon Agents show higher rates of selfrecommendation and greater success compared to baseline agents, indicating enhanced persuasion abilities.

## 6.3 Camouflage

Camouflage is central to Avalon. Evil roles must deceive loyal servants while subtly sabotaging missions. Skilled players create elaborate lies and misdirection. Loyal servants also engage in camouflage to conceal their identities, especially when under suspicion.

In Figure 4, the rates of various behaviors exhibited by AI agents are displayed. Notably, the agents display a notably high tendency to reveal their identities at the commencement of the game, particularly among the roles associated with the good side. Intriguingly, in the roles of Morgana and Assassin, agents opt to either conceal or assume different identities without explicit instructions to do so in the initial strategy. Specifically, Morgana and the Assassin display rates of assuming alternate identities of 10% and 15%, respectively, a strategy akin to that observed in human players, where Percival perceives both Merlin and Morgana but lacks precise knowledge of their identities. This spontaneous adoption of deceptive behaviors by AI agents stands out as a captivating observation, underscoring their adaptability and strategic acumen in the pursuit of game victory.

![](images/ea8e0dba268bb4f07f3d83d2af46b19e4fbc3047e9cb126bff3e84b9cede8676.jpg)

![](images/5253db125311da613d1b3cf9b09ce2567fbd0a0d4b5e23d5314316e4680da323.jpg)

![](images/8bebbb86bd255de2e59d7c76fd0573fb7240a08a53dabd401078a44f5f03be10.jpg)

Figure 5: The teamwork and confrontation behaviors when playing different roles. Each subfigure shows the attitude distribution of the player portraying specific role (on the top) towards players in other roles (on the left).  
![](images/84a386d89a653b05429a71c5015107c6c25cf7ddbf35236e05ef268d8f90ed5f.jpg)

![](images/7dbc109c56ffe95d713fbbedb3d8d375f90d49f51410132e20ce5d5d594204b0.jpg)

![](images/917e665b99fd778d8b8261e32289b5de8d31195ad8a3fcffda468c60583d14b4.jpg)  
Figure 6: (a): The sharing behavior when playing Percival and Merlin at the first round. (b) and (c): The teamwork vacillation between different rounds.

## 6.4 Teamwork and Confrontation

Teamwork is vital for loyal servants to identify each other and succeed in missions by strategizing, discussing assignments, and sharing information to uncover evil roles. Confrontations arise when suspicions lead to accusations, resulting in intense exchanges where accusers present reasoning and the accused offer defenses or deflect suspicion onto others.

In Figure 5 (a), teamwork and confrontation rates of good side roles are depicted. Loyal Servants tend to avoid confrontation due to their lack of specific identity information. However, Merlin, aware of Morgana and Assassin, confronts them frequently. Percival, aware of Merlin and Morgana without knowing their exact identities, confronts both. These observations highlight the adaptive strategies of AI agents, mirroring the social dynamics of human players in Avalon.

Figure 5 (b) shows teamwork and confrontation rates of baseline agents. Rates remain consistent across roles, suggesting they do not adjust strategies based on role assumptions.

## 6.5 Sharing

Sharing is essential for Percival and Merlin. They possess more information than other good roles, and sharing their insights aids in winning the game.

However, excessive sharing of known information may also benefit the opposing side, as discussions are public to all players. Therefore, strategic sharing of information is necessary to win the game.

Figure 6 (a) depicts the proportion of known information shared with other players by different agents playing the roles of Merlin and Percival in the first round of the game. It is observed that both the agents designed by us and the baseline agents exhibit an excessive level of sharing behaviors.

## 6.6 Vacillation

At the game’s onset, some players possess identity clues, like Percival knowing Morgana and Merlin without distinction, while others, like Loyal Servants, lack such info. Both situations require players to deduce identities for their camp’s benefit. Analyzing teamwork proportions across rounds reveals players’ ability to discern allies and foes.

Figure 6 (b) illustrates Loyal Servants’ teamwork tendencies, while (c) shows Percival’s tendencies towards Morgana and Merlin. Throughout the game, players increasingly collaborate with teammates and less with enemies. However, Loyal Servants face greater challenges inferring roles, leading to higher teamwork with potential foes.

## 6.7 Behavior Spontaneity

Teamwork and confrontation behaviors of players arise spontaneously due to game mechanics fostering interaction and competition. Teamwork aids in identifying evil roles, facilitating successful quests. However, teamwork often brings confrontation, as doubts about role identities persist. Even without strategic learning mechanisms, players exhibit these behaviors, showing their spontaneous nature. However, behavior distributions vary significantly

between agents with and without strategic learning.   
The relevant analysis is provided at the Section D.

## 7 Conclusion

This paper explores the social behaviors of LLMbased agents in the Avalon game. We introduce a multi-agent framework facilitating efficient communication and interaction. This framework includes memory, analysis, planning, action, and response modules capable of learning from experience. Unlike prior studies, our research delves into the social dynamics of these agents in gameplay scenarios. Our evaluation showcases the success of our framework in achieving winning strategies and the adaptability of LLM agents in complex social interactions. Future work involves optimizing our approach, exploring its applicability in diverse game environments, and further understanding LLM agents’ potential in dynamic social interactions.

## 8 Limitations

Although the LLM agent framework we proposed has performed well in the Avalon game, there are also limitations of high cost and slow interaction speed, due to multiple accesses to the model required for each interaction. Additionally, from the behaviors exhibited by the agent, there are also instances of unreasonable behavior distribution, such as excessive self-disclosure actions. In the future, we will explore and improve these aspects.

## Acknowledgements

This research is supported, in part, by SMP-IDATA Open Youth Fund. This research is supported, in part, by the National Key R&D Program of China (Grant No.2023YFF0725001), National Natural Science Foundation of China (Grant No.92370204), Guangzhou-HKUST(GZ) Joint Funding Program (Grant No.2023A03J0008), Education Bureau of Guangzhou Municipality.

## References

Elif Akata, Lion Schulz, Julian Coda-Forno, Seong Joon Oh, Matthias Bethge, and Eric Schulz. 2023. Playing repeated games with large language models. ArXiv, abs/2305.16867.

Nicolo’ Brandizzi, Davide Grossi, and Luca Iocchi. 2021. Rlupus: Cooperation through emergent communication in the werewolf social deduction game. ArXiv, abs/2106.05018.

Zhendong Chen, Siu Cheung Hui, Fuzhen Zhuang, Lejian Liao, Fei Li, Meihuizi Jia, and Jiaqi Li. 2022. Evidencenet: Evidence fusion network for fact verification. In Proceedings of the ACM Web Conference 2022, pages 2636–2645.

Yao Fu, Hao Peng, Tushar Khot, and Mirella Lapata. 2023. Improving language model negotiation with self-play and in-context learning from ai feedback.

Chen Gao, Xiaochong Lan, Zhi jie Lu, Jinzhu Mao, Jing Piao, Huandong Wang, Depeng Jin, and Yong Li. 2023. S3: Social-network simulation system with large language model-empowered agents. ArXiv, abs/2307.14984.

Navid Ghaffarzadegan, Aritra Majumdar, Ross Williams, and Niyousha Hosseinichimeh. 2023. Generative agent-based modeling: Unveiling social system dynamics through coupling mechanistic models with generative artificial intelligence. ArXiv, abs/2309.11456.

Yuya Hirata, Michimasa Inaba, Kenichi Takahashi, Fujio Toriumi, Hirotaka Osawa, Daisuke Katagami, and Kousuke Shinoda. 2016. Werewolf game modeling using action probabilities based on play log analysis. In Computers and Games.

Zhao Kaiya, Michelangelo Naim, Jovana Kondic, Manuel Cortes, Jiaxin Ge, Shuying Luo, Guangyu Robert Yang, and Andrew Ahn. 2023. Lyfe agents: Generative agents for low-cost real-time social interactions.

Enkelejda Kasneci, Kathrin Seßler, Stefan Küchemann, Maria Bannert, Daryna Dementieva, Frank Fischer, Urs Gasser, Georg Groh, Stephan Günnemann, Eyke Hüllermeier, et al. 2023. Chatgpt for good? on opportunities and challenges of large language models for education. Learning and individual differences, 103:102274.

Sharon Levy, Robert E Kraut, Jane A Yu, Kristen M Altenburger, and Yi-Chia Wang. 2022. Understanding conflicts in online conversations. In Proceedings of the ACM Web Conference 2022, pages 2592–2602.

Paul Pu Liang, Jeffrey Chen, Ruslan Salakhutdinov, Louis-Philippe Morency, and Satwik Kottur. 2020. On emergent communication in competitive multiagent teams. ArXiv, abs/2003.01848.

Bill Yuchen Lin, Yicheng Fu, Karina Yang, Prithviraj Ammanabrolu, Faeze Brahman, Shiyu Huang, Chandra Bhagavatula, Yejin Choi, and Xiang Ren. 2023. Swiftsage: A generative agent with fast and slow thinking for complex interactive tasks. ArXiv, abs/2305.17390.

Yang Liu, Yuanshun Yao, Jean-Francois Ton, Xiaoying Zhang, Ruocheng Guo, Hao Cheng, Yegor Klochkov, Muhammad Faaiz Taufiq, and Hanguang Li. 2023. Trustworthy llms: a survey and guideline for evaluating large language models’ alignment. ArXiv, abs/2308.05374.

Rajiv Movva, S. Balachandar, Kenny Peng, Gabriel Agostini, Nikhil Garg, and Emma Pierson. 2023. Large language models shape and are shaped by society: A survey of arxiv publication patterns. ArXiv, abs/2307.10700.

Noritsugu Nakamura, Michimasa Inaba, Kenichi Takahashi, Fujio Toriumi, Hirotaka Osawa, Daisuke Katagami, and Kousuke Shinoda. 2016. Constructing a human-like agent for the werewolf game using a psychological model based multiple perspectives. 2016 IEEE Symposium Series on Computational Intelligence (SSCI), pages 1–8.

Joon Sung Park, Joseph C. O’Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. 2023. Generative agents: Interactive simulacra of human behavior.

Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, and Jianfeng Gao. 2023. Instruction tuning with gpt-4. arXiv preprint arXiv:2304.03277.

Chen Qian, Xin Cong, Cheng Yang, Weize Chen, Yusheng Su, Juyuan Xu, Zhiyuan Liu, and Maosong Sun. 2023. Communicative agents for software development. ArXiv, abs/2307.07924.

Zijing Shi, Meng Fang, Shunfeng Zheng, Shilong Deng, Ling Chen, and Yali Du. 2023. Cooperation on the fly: Exploring language agents for ad hoc teamwork in the avalon game.

Qiurong Song and Jiepu Jiang. 2022. How misinformation density affects health information search. In Proceedings of the ACM Web Conference 2022, pages 2668–2677.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Chen Feng Tsai, Xiaochen Zhou, Sierra S Liu, Jing Li, Mo Yu, and Hongyuan Mei. 2023. Can large language models play text games well? current state-of-the-art and open questions. arXiv preprint arXiv:2304.02868.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems, 30.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. 2023a. Voyager: An open-ended embodied agent with large language models.

Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. 2023b. A survey on large language model based autonomous agents. arXiv preprint arXiv:2308.11432.

Shenzhi Wang, Chang Liu, Zilong Zheng, Siyuan Qi, Shuo Chen, Qisen Yang, Andrew Zhao, Chaofei Wang, Shiji Song, and Gao Huang. 2023c. Avalon’s game of thoughts: Battle against deception through recursive contemplation.

Tianhe Wang and Tomoyuki Kaneko. 2018. Application of deep reinforcement learning in werewolf game agents. 2018 Conference on Technologies and Applications ofArtificial Intelligence (TAAI), pages 28–33.

Yufei Wang, Wanjun Zhong, Liangyou Li, Fei Mi, Xingshan Zeng, Wenyong Huang, Lifeng Shang, Xin Jiang, and Qun Liu. 2023d. Aligning large language models with human: A survey. ArXiv, abs/2307.12966.

Sarah Wiseman and Kevin B. Lewis. 2019. What data do players rely on in social deduction games? Extended Abstracts of the Annual Symposium on Computer-Human Interaction in Play Companion Extended Abstracts.

Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, et al. 2023. The rise and potential of large language model based agents: A survey. arXiv preprint arXiv:2309.07864.

Yuzhuang Xu, Shuo Wang, Peng Li, Fuwen Luo, Xiaolong Wang, Weidong Liu, and Yang Liu. 2023a. Exploring large language models for communication games: An empirical study on werewolf.

Zelai Xu, Chao Yu, Fei Fang, Yu Wang, and Yi Wu. 2023b. Language agents with reinforcement learning for strategic play in the werewolf game.

Kai-Cheng Yang and Filippo Menczer. 2023. Anatomy of an ai-powered malicious social botnet. ArXiv, abs/2307.16336.

Haoqi Yuan, Chi Zhang, Hongcheng Wang, Feiyang Xie, Penglin Cai, Hao Dong, and Zongqing Lu. 2023. Plan4mc: Skill reinforcement learning and planning for open-world minecraft tasks.

Xuanhe Zhou, Guoliang Li, and Zhiyuan Liu. 2023. Llm as dba.

Xizhou Zhu, Yuntao Chen, Hao Tian, Chenxin Tao, Weijie Su, Chenyu Yang, Gao Huang, Bin Li, Lewei Lu, Xiaogang Wang, Yu Qiao, Zhaoxiang Zhang, and Jifeng Dai. 2023. Ghost in the minecraft: Generally capable agents for open-world environments via large language models with text-based knowledge and memory.

## A Appendix

## A.1 Avalon Introduction

Avalon is designed for 5 to 10 players. Specifically, we focus on the 6-player variant of the game.

Player roles. Roles including Merlin, Percival, Morgana, Assassin, and two Loyal Servants, are divided into good and evil sides. Merlin, Percival, and loyal servants are on the good side, while Morgana and Assassin are on the evil side. Players are assigned roles secretly, with some having special abilities. Morgana and Assassin are initially aware of each other. Percival is able to see Merlin and Morgana but does not know their exact identities. Merlin is aware of the identities on the evil side.

Quest team assignment. After receiving roles, players engage in 3-5 rounds of discussion and voting for a certain number of players to form a quest team. At the start of each round, a leader is assigned in rotation. The leader hosts a discussion, followed by a public vote on quest team members. If more than half of the votes agree, the team forms; otherwise, leadership rotates to the next player for further discussion and voting. Each round allows up to five discussion and voting cycles, with the leader directly assigning team members after the fifth round.

Quest phase. The quest outcome is determined by the cards submitted by the quest team. Good players can only submit success cards, while evil players can choose to submit either success or failure cards. A quest is successful if all team members vote for success, and fails if one or more members vote for failure.

End of the game. The game ends when three quests succeed (good side wins) or three quests fail (evil side wins). Additionally, the evil players can win by correctly identifying Merlin at the end.

## A.2 Game Rules and Role Description

You are playing a game called the Avalon with some other players. This game is based on text conversations. Here are the game rules: Roles: The moderator is also host, he organised this game and you need to answer his instructions correctly. Don’t talk with the moderator. There are five roles in the game, Merlin, Percival, Loyal Servant,

Morgana, Assassin. Merlin, Percival and Loyal Servant belong to good side and Morgana and Assassin belong to evil side There are two alternate phases in this game, reveal phase and quest phase. When it’s reveal phase: You need follow the instructions of the moderator. You needn’t worry about other players and moderator knowing what you say and do. No need to worry about suspicions from others during the phase. If you are Merlin, you can know which two players are Morgana and Assassin but you can’t know which one is Morgana or Assassin specifically. If you are Percival, you can know which two players are Merlin and Morgana but you can’t know which one is Merlin or Morgana specifically. If you are Morgana, you can know which player is Assassin.If you are Assassin, you can know which player is Morgana. If you are Loyal Servant, you can’t get any information in this phase. The quest phase includes 5 rounds. A round includes discussion, voting and execute quest. At every round, all players need to discuss about which players (candidates) will execute the quest at current round. And then all players need to vote if the candidates should execute the quest, if the agreement exceeds 1/2, the candidates will execute the quest, otherwise, discuss again and vote again. When executing quest, the candidates need to choose to make quest successful or failed. If all candidates choose to make quest successful, the quest will succeed. If anyone makes the quest failed, the quest will fail. At the end of a round, if the

quest succeed, good side will get one point, otherwise, evil side will get one point. Which side get 3 points earlier, which side wins the game. If you are Assassin, at the end of a round, you can choose to identify which one is Merlin, if the identifying is successful, the red camp directly win the game. If not successful, the Assassin will expose his identification. Objectives: your goal is to help your side get 3 points and win the game. If you are Assassin, you also need to reason which player is Merlin as early as possible. Tips: To complete the objective: you should analyze and use your ability correctly. During quest phase, you need to reason carefully about the roles of other players and be careful not to reveal your own role casually unless you’re cheating other players. Only give the player’s name when making a decision/vote, and don’t generate other players’ conversation. Reasoning based on facts you have observed and you cannot perceive information (such as acoustic info) other than text. You are {player}, the {role}. You’re playing with 5 other players. Do not pretend you are other players or the moderator. Always end your response with ‘<EOS>’.

## A.3 Module Prompts

Our designed prompts for different modules are presented in Tables 4 and 5.

## A.4 Heuristic Rules for LLM Gameplay

In the gameplay, we used LLM to extract information from the responses of the agents. For example, when the agent selects a player, it extracts the player number, and when voting, it extracts the player’s voting result. With several demonstrations of how to extract corresponding information, LLM can extract information very accurately to help the game proceed smoothly. Table 3 shows some cases of extraction.

It is observed agents sometimes may fail to answer questions correctly, such as voting with unclear attitudes. In order to allow the game to proceed smoothly, we design the following heuristic rules. When voting for quest candidates, if the agent’s answer is unclear, we assume that it agrees. When voting the quest for success or failure, if the agent’s answer is unclear, we default to it voting for failure. When agents select an excessive number of players, we truncate the selection to meet the quest’s requirements. In cases where the agents choose too few players, the host will repeat question to the agent. If the required player count is still not met even after multiple retries, the program steps in to assist by making a random selection on behalf of the agent.

## A.5 Ablation Study

To validate the efficacy of the proposed modules, we conducted an ablation study under both with and without learning from experience setting. Initially, we assessed the effectiveness of the Improving Strategy Module (IS), the Analysis of Others’ Strategies Module (AO), and the Analysis Module (AM) within the context of the learning from experience setting, wherein strategies were updated based on accumulated gameplay for both our agents and the baseline agents. In this evaluation, the proposed agents engaged in ten games, assuming evil side roles, against the baseline agents for each module. Following these games, the wining rate (WR), quest engagement rate (QER), and the failure voting rate (FVR) were measured and reported for analysis. Table 6 presents the outcomes of the ablation study conducted within the learning-fromexperience setting. It is discernible that in the absence of the Improving Strategy module, where the strategy remains static but the agent can still glean insights from other players’ strategies, the winning rate decreases by 20%. Additionally, the agents exhibit reduced aggression, indicated by lower quest engagement rates and failure voting rates. Furthermore, the absence of the Analysis of Others’ Strategies module and the Analysis Module also leads to a decline in the winning rate. In these scenarios, the agents adopt a cautious gameplay approach, resulting in significantly lower quest engagement rates but higher failure voting rates.

![](images/79e0fa66ee26bfc9e133b7592aa95b908fdcc91fa242e5d1d88f1b90afac2c5c.jpg)  
Table 3: Cases of LLM-based extraction

Summarization:   
Within the context of the Avalon game, please assist {Player i} in   
summarizing the conversations known to him from the current phase. These   
conversations are structured in JSON format, with “message” signifying   
the content of the conversation, "name" identifying the speaker, and   
“message\_type” indicating the type of message relevant to {Player i}.   
Specifically,“public” implies that all players have access to the message,   
while “private” implies that only {Player i} has access to it.   
Conversations: {conversations}.   
Analysis:   
Your task is to analyze roles and strategies of the players who might be   
your enemies according to their behaviors. The analysis should be no more   
than 100 words. The behaviors are summarized in paragraphs.   
Your name is {Name} your role is {Role}.   
The summary is {Summary}.   
Planning:   
Your task is to devise a playing plan that remains in harmony with your   
game goal and existing strategy, while also incorporating insights from your   
previous plan and current environment state.   
{Role Information}   
Goal: {Goal}   
Strategy: {Strategy}   
Your previous plan: {Plan}   
Summary of previous rounds: {Summary}   
Analysis about other players: {Analysis}.   
Action:   
Your objective is to make decisions based on your role, your game goal   
and the current game state. There are five types of actions you can take:   
choosing players, voting (agree or disagree), performing missions (make   
missions succeed or fail), using non-verbal signals (raise hands up, put   
hands down, open eyes, or close eyes), and choosing to remain silent. Only   
one action type can be selected at a time. If you decide to choose players,   
you can choose multiple players according to Host’s question.   
{Role Information}   
Goal: {Goal}   
Strategy: {Strategy}   
Your current plan: {Plan}   
Summary of previous rounds: {Summary}   
Analysis about other players: {Analysis}.   
Host’s Instruction: {Instruction}.   
Response:   
Your task is to provide detailed response to the question of Host, in   
accordance with the provided actions. Your response should be no more than   
100 words.   
{Role Information}   
Goal: {Goal}   
Strategy: {Strategy}   
Your current plan: {Plan}   
Summary of previous rounds: {Summary}   
Host’s Instruction: {Instruction}.   
current actions: {actions}  
Table 4: Input prompts of our proposed different modules.

![](images/0151eb4a91ca4c58f37864bf4c5627c4a7774761cf93a9d08e760dbde6fbbfe8.jpg)  
Table 5: Input prompts of our experience learning module.

<table><tr><td rowspan="2">Method</td><td rowspan="2">WR(%)</td><td colspan="2">QER(%)</td><td colspan="2">FVR(%)</td></tr><tr><td>Morgana</td><td>Assassin</td><td>Morgana</td><td>Assassin</td></tr><tr><td>full</td><td>80</td><td>44.1</td><td>49.1</td><td>66.6</td><td>78.5</td></tr><tr><td>w/o. IS</td><td>60</td><td>42.8</td><td>39.3</td><td>46.1</td><td>100</td></tr><tr><td>w/o. AO</td><td>70</td><td>18.3</td><td>8.3</td><td>100</td><td>100</td></tr><tr><td>w/o. AM</td><td>50</td><td>29.3</td><td>39</td><td>87.5</td><td>100</td></tr></table>

Table 6: Ablation Study on Experience Learning: Compare of full framework, without improving strategy (IS), without analysis strategies of others (AO) and without analysis module (AM).
<table><tr><td rowspan="2">Method</td><td rowspan="2">WR(%)</td><td colspan="2">QER(%)</td><td colspan="2">FVR(%)</td></tr><tr><td>Morgana</td><td>Assassin</td><td>Morgana</td><td>Assassin</td></tr><tr><td>all modules</td><td>90</td><td>55.5</td><td>58.3</td><td>93.7</td><td>100</td></tr><tr><td>w/o analysis</td><td>80</td><td>44.1</td><td>47.5</td><td>100</td><td>100</td></tr><tr><td>w/o. plan</td><td>60</td><td>55</td><td>16.6</td><td>90</td><td>100</td></tr><tr><td>w/o. action</td><td>80</td><td>45.6</td><td>45.6</td><td>100</td><td>100</td></tr></table>

Table 7: Module Ablation: under the setting without learning from experience.

Following the initial evaluation, we proceeded to assess the effectiveness of the Analysis Module, Planning Module, and Action Module under conditions where learning from experience was not incorporated. In this scenario, strategies were not updated for both our agents and the baseline agent. It is essential to note that the games were conducted independently, with no influence from previous games on future gameplay. Table 7 presents the results from the module ablation study conducted without incorporating learning from experience. It is discernible that the absence of the planning module results in a notable 20% decrease in the winning rate. Additionally, the Assassin exhibits a significantly lower quest engagement rate, indicating a tendency to overlook the mission objective without the guidance of a strategic plan. This underscores the critical importance of the planning module in ensuring that agents consistently progress toward winning the game.Furthermore, in the absence of both the analysis and action modules, the agents exhibit a slightly lower quest engagement rate. Despite this, they manage to maintain an impressive 80% winning rate.

In the final phase of our evaluation, we scruti-
<table><tr><td rowspan="2">Method</td><td rowspan="2">WR(%)</td><td colspan="2">QER(%)</td><td colspan="2">FVR(%)</td></tr><tr><td>Morgana</td><td>Assassin</td><td>Morgana</td><td>Assassin</td></tr><tr><td>all players</td><td>90</td><td>55.5</td><td>58.3</td><td>93.7</td><td>100</td></tr><tr><td>teammates only</td><td>80</td><td>26.8</td><td>48.1</td><td>62.5</td><td>100</td></tr><tr><td>adversaries only</td><td>90</td><td>38.3</td><td>45.3</td><td>92.3</td><td>100</td></tr></table>

Table 8: Analysis Module Ablation: under the setting without learning from experience. Analyzing different objects.

![](images/54e762351c680c0357daef2dc17d1c200dd76629112c240ce638ab4fe0f2dad8.jpg)  
Figure 7: Persuasion example

![](images/7f178f0ccfbac6dca47af77db22a85cbb965e60aea07d7cb13f54ff19c645b85.jpg)  
Figure 8: Camouflage example

nized the impact of analysis on all players, teammates and adversaries. In each configuration, our agents assumed the roles of the evil side in ten games, facing off against baseline agents aided by corresponding analysis information. The results, encompassing winning rate, quest engagement rate, and failure voting rate, are tabulated in Table 8. It becomes apparent that when analysis information is restricted solely to teammates, the winning rate declines by 10%. In response, our proposed AI agents adopt a less aggressive approach, evident in reduced quest engagement rates and failure voting ratings. However, when analysis information pertains exclusively to adversaries, there is a decrease in quest engagement rates while retaining the winning rate and failure voting rate. This phenomenon can be attributed to the strategic advantage gained by the Assassin, who can identify Merlin with the aid of analysis information on adversaries. Consequently, the analysis of adversaries proves to be paramount for the evil side’s victory in Avalon games for AI agents.

## B Case Study

In Figures 7, 8, 9 and 10, we present examples to show how the AI agents perform the social behaviors in the Avalon games.

## C Exploration on LLaMA-Based Agents

For broader validation, we implemented our framework on the Llama2-7b-chat-hf model. However, LLaMA-based agents face constraints due to the model’s language understanding capabilities and token limitations. Preliminary exploration without further analysis is discussed below.

<table><tr><td rowspan="2">Base Model</td><td colspan="6">VRR (%)</td></tr><tr><td>Loyal Servant</td><td>Merlin</td><td>Percival</td><td>Morgana</td><td>Assassin</td><td>Average</td></tr><tr><td>LLaMA2</td><td>51.9</td><td>61.0</td><td>53.6</td><td>66.5</td><td>66.9</td><td>59.9</td></tr><tr><td>GPT-3.5</td><td>81.7</td><td>84.2</td><td>81.9</td><td>89.7</td><td>87.6</td><td>85.0</td></tr></table>

Table 9: Valid Response Rate (VRR) of different models <sub>143</sub>

![](images/43387c7bb078a9db59bad9909ce59bf7ecf5f08b3b77e57d973f62cd4faf8abe.jpg)  
Figure 10: Leadership example

Table 9 presents the performance of agents based on LLaMA2 in the Avalon game, where we measure their performance using Valid Response Rate (defined in equation 8). Compared to GPT3.5, LLaMA shows a decrease of 25.1% in this metric. This could be attributed to LLaMA’s poorer language comprehension abilities compared to GPT3.5, resulting in its inability to grasp the complex content of the Avalon game.

Valid Response Rate (VRR). Agents are required to engage in discussion, select players, and vote. A Valid Response is defined as a response that adheres to these requirements. the VRR is calculated as follows:

$$
V R R = ( \frac { \# V a l i d R e s p o n s e s } { \# T o t a l ~ R e s p o n s e s } ) \times 1 0 0 \%\tag{8}
$$

## D Teamwork and Confrontation

Figure 11 and Figure 12 illustrate the differences in teamwork and confrontation behaviors of agents under conditions with and without experience learning.

Figure 12 shows that, without strategic learning, evil-side players (e.g., Morgana) overly confront, while good-side players confront less, with minimal variation. This contrasts with Figure 11, depicting agents with strategic learning. Here, the introduction of strategic learning mitigates excessive confrontation by evil-side players, who strategically engage in more teamwork. Conversely, good-side players strategically increase confrontation with potential enemies while reducing it with potential teammates.

![](images/40a3447928f30ecfa637da09d2645d958f9ee02c0bcbabf687bd50baa153407a.jpg)  
Figure 11: The teamwork and confrontation behaviors when playing different roles: each subfigure shows the attitude distribution of the player portraying specific role (on the top) towards players in other roles (on the left).

![](images/1dd43c1a41b0432a8e42eed29fc215c8b32dfce64cbf25600e841ee9ddb1402f.jpg)  
Figure 12: The teamwork and confrontation behaviors when playing different roles (agents without experience learning module)