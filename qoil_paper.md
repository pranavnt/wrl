# Q-OIL: Optimistic Exploration in Learning from Human Interventions

**Anonymous Author(s)**
Affiliation
Address
email

*Submitted to the 10th Conference on Robot Learning (CoRL 2026). Do not distribute.*
*Project website: https://sites.google.com/view/qoil*

---

## Abstract

As robots are deployed in the real world, they will inevitably make mistakes. Human interventions provide valuable feedback for improving their behavior, enabling them to adapt online to new scenarios. However, studying human-in-the-loop robot learning is difficult: real-world experiments are slow and expensive, so most algorithm development relies on simulated human models. We show that common human modeling assumptions do not match how real people intervene in practice. Through an analysis of intervention data from human participants, we find that humans tend to intervene when agent progress stalls, and that their corrective actions are often suboptimal. Motivated by these findings, we introduce a progress-based model of human intervention that achieves higher rank correlation with real-user algorithm rankings, enabling realistic algorithm benchmarking and development in simulation. Driven by these insights, we identify shortcomings in prior methods for learning from interventions and propose **Q-OIL**, a method that introduces optimistic bonuses on intervention transitions in reinforcement learning, encouraging the agent to revisit and learn from informative intervention states. We show that with the right design choices, including decoupled critics to avoid over-optimism and imitation-learning regularization, our proposed method learns more efficiently than competitive baselines, achieving 10–50% higher success in simulation and on real-world robotic tasks. By grounding both evaluation and learning in realistic human behavior, this work provides a practical path toward scalable robot learning from interventions.

**Keywords:** Human-in-the-loop, Robot Learning, Reinforcement Learning

---

## 1 Introduction

Robot policies trained offline on large-scale datasets are brittle [1, 2], often struggling to generalize beyond their training distributions [3] and failing across diverse testing conditions. For robust deployment, robots must be able to adapt online from feedback [4, 5, 6, 7]. As robots fail around human observers, we can rely on human feedback through corrections [8, 9, 10], where humans intervene when the robot gets stuck or makes mistakes. Such signals arise organically in the real world; for example, users may physically steer a manipulator toward the goal [11, 12, 7] or take over in self-driving systems [13]. Interventions can be viewed as more than resets or recovery actions [13]. Corrective actions convey implicit preference signals, i.e., information about failures in the robot's behavior and potential ways to improve [14, 15, 16, 17, 18, 19, 20, 21].

Despite recent progress in learning from interventions, algorithm development remains difficult because real-world interaction and evaluation are slow and expensive. As a result, development relies on synthetic human models [22, 17] which enable repeatable evaluation and systematically guide algorithmic components (e.g., how to incorporate corrections, the exploration-safety trade-off, robustness to noise) before deployment. However, common synthetic human models and learning algorithms are insufficiently grounded in real-world behavior. They make assumptions that we show are not reflective of *real human intervention data*. This creates a gap between methods on synthetic human models versus their performance when exposed to real users. What is needed is a principled understanding of *when* and *how* humans intervene, both to guide our algorithmic design choices and to evaluate these methods. We first analyze real human interventions in robot manipulation. Our analysis suggests both how to better model human interventions and how to learn from them.

> **Figure 1: Overview of Q-OIL.** Real-user data shows that humans intervene when robot progress stalls and provide suboptimal corrections. Q-OIL treats interventions as optimistic exploration bonuses while preventing bonuses from propagating to pre-intervention states to support stable real-world learning.

First, we find that rather than reacting to per-step suboptimality or failures, humans intervene when the robot's progress over a horizon stagnates below a threshold. This pattern aligns with cognitive models of caregiving: humans act on the agent's ability over time rather than on isolated states [23]. Given this finding, we propose a progress-based synthetic intervention model that aggregates value improvements over a horizon and better matches observed humans, yielding algorithm rankings more consistent with real human evaluation (see Fig. 4).

Second, this analysis shows that real corrections are suboptimal and can diverge from the learner policy during training, which suggests that algorithms for learning from interventions must be robust to noisy, divergent interventions, i.e., use feedback to adapt robot behavior without compromising asymptotic performance through over-reliance. This motivates our key algorithmic insight: use provided interventions as optimistic guidance for exploration, rather than optimal supervision targets. This speeds up training without being catastrophically misled by suboptimal behavior.

To incorporate this, we introduce **Q-OIL** (**Q-O**ptimistic **I**ntervention **L**earning), a simple online RL framework that provides an additional positive reward, i.e., optimism bonus, to intervention transitions. Intuitively, this softly biases exploration to regions that humans guide the robot toward, which are more likely to contain successful behaviors. The optimism influence decays as interventions become less frequent, shifting learning back to the original task reward. Q-OIL handles diverse users and suboptimal or policy-divergent interventions, rather than requiring perfectly optimal, consistent demonstrations. To make this stable, we introduce a critical decoupling of optimism from TD learning [24]: the optimistic critic drives exploration and policy updates, while a task-only critic provides TD targets, preventing catastrophic bonus propagation to non-intervention states. We show the efficacy of Q-OIL in robotics problems both in simulation and in the real world, with real users.

In summary, we make the following contributions:

1. We analyze human intervention data to ground synthetic human models for realistic simulation evaluation and informed algorithm design. Our findings show that interventions occur when robot progress stagnates, motivating a progress-based model that better matches human behavior.
2. Based on human analysis, we propose Q-OIL, a novel RL method that treats interventions as optimistic guidance for exploration rather than optimal supervision targets.
3. Across diverse real and simulated robot manipulation tasks with human participants, Q-OIL learns faster, reduces human effort, and remains robust to suboptimal and diverse interventions.

---

## 2 Related Work

**Learning from Human Demonstrations.** Behavior cloning from offline demonstrations [25, 26, 27] is a standard approach for training robot policies, but suffers from compounding errors under distribution shift at deployment [5, 20]. Reinforcement learning can learn robust behaviors from task rewards [28, 29], but is often sample-inefficient due to undirected exploration. Recent works [30, 31, 32, 6, 33] warm-start RL with offline datasets, and particularly, Lu et al. [34] combine behavior cloning with off-policy actor-critic updates. However, collecting task demonstrations is costly. In contrast, we leverage easy-to-provide online interventions [8] to bias policy behavior during training.

**Interactive Learning.** Interactive learning methods [5, 8, 9, 35] address the drawbacks of offline approaches by collecting online feedback, reducing compounding error from quadratic to linear regret [5]. A natural class of methods allows humans to override policy behavior with corrective actions [8, 12, 16, 35], which prior work uses as behavior-cloning targets [8, 12], policy constraints [13, 21], or human-behavior models for RL [11, 35, 36]. However, these methods make strong assumptions about human interventions. While Luo et al. [17] treat interventions as negative feedback, leading to pessimistic and inefficient behavior, Q-OIL is inspired by auxiliary reward-driven exploration [37, 38] and adds optimistic rewards to intervention transitions. Notably, Luo et al. [35] makes no assumptions on human behavior and adds interventions to the replay buffer, achieving strong real-world results; however, we show it is inefficient relative to Q-OIL.

**Modeling Human Behavior.** Learning from interventions requires accurate models of when and how humans intervene. Because real-world training is costly [6], prior work relies on approximate human models: TAMER [15] learns a scalar reward predictor, while others use Boltzmann-rational models [11, 39], probit triggers [18], or threshold-based rules tied to instantaneous suboptimality [17]. A common assumption is that human corrections are Markovian, i.e., merely reactions to instantaneous suboptimality, which our analysis contradicts. Consistent with cognitive theories of utility trade-offs [40] and computational models of caregiving [23], our proposed model accounts for agent progress over a short horizon when triggering interventions, grounded in real human data.

---

## 3 Problem Statement

We build on the framework of interactive imitation learning, focusing on learning from human interventions [8]. Consider an interactive control setting over an MDP $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \gamma, \rho_0)$, where $\mathcal{S}$ is the state space, $\mathcal{A}$ is the action space, and $\mathcal{T}, \mathcal{R}, \rho_0, \gamma$ are the transition kernel, (sparse) reward function, initial state distribution, and discount, respectively. The objective is to learn a policy $\pi_\theta(a \mid s)$ that maximizes expected discounted return, i.e.,

$$\pi_\theta = \arg\max_\theta \; \mathbb{E}_{\pi_\theta}\!\left[\sum_{t \geq 0} \gamma^t \, r(s_t)\right].$$

During deployment, a human observes the agent and may intervene. We model the human as a decision function $H = (g, \pi_h)$ with two components: a gating function $g$ that decides *when* to intervene, and a human policy $\pi_h$ that decides *how* to intervene. To allow temporal context, both may depend on a recent history $\tau_{t-L:t} = (s_{t-L}, a_{t-L}, \ldots, s_t)$, so $g : \mathcal{S}^{L+1} \times \mathcal{A}^L \to [0, 1]$ outputs the probability of intervening at time $t$, and $\pi_h : \mathcal{S}^{L+1} \times \mathcal{A}^L \to \mathcal{A}$ returns a distribution over actions. The resulting rollout policy is the mixture

$$\pi'(a \mid s_t, \tau_{t-L:t}) = g(\tau_{t-L:t})\,\pi_h(a \mid \tau_{t-L:t}) + \big(1 - g(\tau_{t-L:t})\big)\,\pi_\theta(a \mid s_t).$$

Thus, modeling human behavior requires simulating $g$ and $\pi_h$. Interventions can be variable-length: single actions or short segments. In the following sections, we (i) collect and analyze real human interventions to characterize *when* and *how* people intervene (Section 4), (ii) propose a non-Markovian progress-based model for $g$ validated against this data (Section 4.2), and (iii) leverage this analysis to develop a new algorithm addressing key failure modes of prior approaches in learning from interventions (Section 5).

---

## 4 Rethinking Simulated Interventions

Developing methods that learn from interventions requires extensive experiments, but real-user studies are slow to reproduce, making it impractical to explore the full design space in deployment. Thus, researchers rely on simulated human models for scalable training and evaluation to build better learning methods. However, if the human model is misaligned with reality, methods optimized for these models and algorithmic choices inferred from such experiments may not transfer to learning from real users. This motivates a central question: *when and how do humans actually intervene?*

**Prior Approaches to Simulating Interventions.** Prior simulated human models assume access to $(\pi^*, Q^*)$, set $\pi_h \equiv \pi^*$, and define $g$ as a Markovian function of $s$ and $a$. Bajcsy et al. [11] models $g(s_t, a) \propto \exp\{Q^*(s_t, a)\}$ as a Boltzmann-rational function; Korkmaz and Biyik [18] use a probit choice model based on $g(s_t, a) = \Phi\!\big(Q^*(s_t, a) - \mathbb{E}_{a' \sim \pi(\cdot \mid s_t)}[Q^*(s_t, a')] - c\big)$, and Luo et al. [17] models $g$ based on action suboptimality, i.e., $g(s_t, a_t) = \mathbb{1}\!\big[Q^*(s_t, a_t^*) - Q^*(s_t, a_t) > \tau\big]$, where $\tau, c$ are thresholds and $\Phi$ is the standard normal CDF. A common assumption here is that $g$ is Markovian and $\pi_h$ is near-optimal. Below, we show that human behavior contradicts both assumptions; thus, methods and conclusions based on these models can diverge from experiments with real humans.

**Collecting Real Human Intervention Data.** We ran a pilot study with 8 participants supervising a simulated robot across 3 manipulation tasks, yielding 24 runs and $\sim 200\text{k}$ transitions. Participants provided interventions during training (Section 6), following the setup of Luo et al. [35]. We recorded when takeovers occurred and how users corrected the robot across learning stages. The study was IRB-approved (Appendix D.1). For analysis, we trained task-specific reference policies and value functions, $\pi^*$, $Q^*(s, a)$, and $V^*(s)$, using RLPD [33] with sparse rewards, which let us quantify interventions in terms of optimality, timing, and convergence. (Details in Appendix D.)

### 4.1 Analyzing Real Human Intervention Data

> **Figure 2:** (a) Progress separates interventions better than instantaneous values. (b) Qualitatively, humans intervene when value plateaus or declines.

**Finding 1: Human intervention gating is non-Markovian.**

Typically, the gating function is modeled as Markovian and reactive to the suboptimality of the action, implying that intervention timing depends only on the current state and action. Figure 2a contradicts this: $Q^*(s, a^\pi)/Q^*(s, a^*)$ is nearly identical for intervention vs. non-intervention states, indicating similar instantaneous values. But, when we measure $k$-step progress as $V^*(s_t) - V^*(s_{t-k})$, the three separate value trajectories (Fig. 2b) show that takeovers follow plateaus or drops in $V^*(s_t)$, suggesting stagnation rather than single-step failure. This matches evidence that corrections can lag agent behavior [15]. Thus, Markovian models insufficiently represent human behavior, leading to experimentally invalid conclusions.

> **Figure 3:** Human interventions have $\sim 3\times$ more suboptimality than $\pi^*$.

**Finding 2: Human interventions are suboptimal demonstrations of the target behavior.**

Several methods [41, 12, 8] treat human corrections as demonstrations by assuming $\pi_h$ is near-optimal, often $\pi_h \equiv \pi^*$. To quantify takeover quality, we compare the value of the intervention action value to the expert action at the same state and convert the gap into a suboptimality score, with $\pi^*$ normalized to 1. On our real dataset, interventions are more suboptimal than $\pi^*$ and human demonstrations, under this metric (Appendix B). Figure 2 shows that interventions do not consistently increase $V^*(s_t)$, implying corrections are noisy. These results contradict $\pi_h \equiv \pi^*$ and motivate methods robust to suboptimal corrections.

**Finding 3: Human corrections diverge from the learned policy over training.**

Across the intervention dataset ($\sim 40\text{k}$), we find a systematic mismatch between human corrections and the learner's converged behavior. Using $Q^*$, we compare values between the converged policy ($a^{\text{final}}$) and human actions and observe $Q^*(s, a^{\text{final}}) \geq Q^*(s, a^{\text{human}})$, implying that at the same states, the final policy prefers different actions with higher values. As both have high success ($\sim 70$–$80\%$), the gap is better explained as divergence in solution modes, which could be a result of sparse rewards admitting multiple solutions (e.g., left- vs. right-side grasps). Human supervisors do not observe the full policy so may provide corrections with an alternative strategy. Therefore, treating such corrections as strict supervision can induce undesirable multi-modal behavior, so instead we motivate using interventions as soft guidance for exploration to speed up training.

### 4.2 A Non-Markovian Gating Model for Human Interventions

Motivated by these findings, we define $g$ as a progress-aware model (PAM) of when humans intervene over recent history $\tau_{t-k:t}$:

$$g(\tau_{t-k:t}) = \Pr(\nu_t = 1 \mid \tau_{t-k:t}) =
\begin{cases}
\alpha, & \text{if } V^*(s_t) - V^*(s_{t-k}) < \delta, \\
\beta, & \text{otherwise.}
\end{cases} \tag{1}$$

using $V^*(s_t) - V^*(s_{t-k})$ as $k$-step progress, and where $k$ is the horizon, $\delta > 0$ the progress threshold, and $\alpha, \beta$ capture intervention stochasticity. Thus, interventions are likely when recent value progress stagnates (including failures where $V^*(s_t) - V^*(s_{t-k}) < 0$). We tune $k, \delta, \alpha, \beta$ by grid search (Appendix F) on real human data. To model how humans intervene, i.e., $\pi_h$, we use an intermediate RL checkpoint ($\sim 50$–$70\%$ success), reflecting that corrections are imperfect.

> **Figure 4:** $\rho$ between approximate models and real human algorithm rankings.

**Validation Against Real-Human Rankings.** A useful intervention model should predict which algorithms succeed with real users. We validate $g$ by held-out intervention prediction and ranking agreement with real-human evaluations. Across all runs, we rank 4 baseline learning algorithms (Section 6) by final success under three gating models: our stagnation model, action suboptimality [17], and random model calibrated to the true intervention rate, and compare these rankings to human evaluations using Spearman correlation.

**Our progress-based gating approach is a stronger approximate human model for evaluating robot intervention learning.** Figure 4 shows stronger agreement with real-human rankings ($\rho = 1.0$) than alternative gating models, providing a more reliable proxy for benchmarking algorithms with real users. In Appendix F, we provide further discussion and show that, by predicting human interventions with higher precision and recall, our model aligns more closely with real user behavior.

---

## 5 How Should We Use Human Interventions for Robot Policy Learning?

As outlined in Section 4.1, prior methods for learning robot policies from interventions are based on invalid assumptions about human behavior. Given this, we ask: *how should we use imperfect and noisy human interventions for robot learning?*

> **Figure 5: Overview of our method: Q-OIL.** Critic updates: $Q_{TD} := r^{\text{task}} + \gamma Q_{TD}$ and $Q_{Opt} := r^{\text{task}} + r^{\text{bonus}} + \gamma Q_{TD}$. The task critic bootstraps from itself, while the optimistic critic adds the bonus reward but bootstraps from the task critic; the optimistic critic drives policy improvement.

**Human interventions provide optimistic signals to guide exploration during RL.**

To this end, grounded in our findings about human behavior, we present an approach based on off-policy model-free RL and optimistic exploration [37]: Q-OIL, which adds a positive reward bonus at intervention transitions to increase the value of regions where humans intervened, softly biasing training exploration without forcing strong alignment to potentially noisy and divergent corrections (Findings 2 and 3). Standard off-policy RL learns a critic by using the given rewards and bootstrapping the value of future states. Training a single critic with the optimism bonus and bootstrapped updates would propagate this bonus backward to all predecessor states of an intervention, which are precisely the states where the policy was failing, incorrectly assigning them higher value targets. We address this catastrophic over-optimism via decoupling the critic into an optimistic and task critic, as illustrated in Fig. 5. This is critical, as an optimistic critic (trained on task reward + bonus) bootstraps from the task critic (trained only on the task reward), thereby confining the bonus to intervention states. The optimistic critic improves the policy, while the task critic is an unbiased estimate of the true value function. We include a longer discussion for this in Appendix A. Finally, we show that this human-informed method leads to improved performance with real users and the better-aligned synthetic human model.

### 5.1 Q-Optimistic Intervention Learning

**Algorithm 1**

```
Require: π_θ, π_int, D_π, D_int, bonus b > 0
 1: for trial i = 1 to N do
 2:     // Training decoupled critics
 3:     Train Q_TD on D_π via L_TD
 4:     Train Q_Opt on D_π via L_Opt
 5:     // Policy improvement
 6:     Update π_θ on D_π and D_int via L_θ
 7:     for timestep t = 1 to T do
 8:         // rollout π_θ
 9:         if π_int intervened at t - 1 then
10:             r^bonus_{t-1} = b
11:             append (s_{t-1}, a^int_{t-1}) to D_int
12:         else
13:             r^bonus_{t-1} = 0
14:         end if
15:         add (s_{t-1}, a_{t-1}, r^task_{t-1}, r^bonus_{t-1}, s_t) to D_π
16:     end for
17: end for
```

Practically, we learn a policy $\pi_\theta$, task critic $Q_{TD}$, and optimistic critic $Q_{Opt}$ using an online off-policy actor–critic algorithm [28, 33], though in principle it may improve any RL method. In Algorithm 1, $\pi_\theta$ is rolled out to interact with the environment, while a human or an approximate model supervisor intervenes with corrective actions. $D_\pi$ stores transitions $(s_t, a_t, r^{\text{task}}_t, s_{t+1})$ and bonus reward $r^{\text{bonus}}_t = b \cdot \mathbb{1}(\text{intervene})$, while $D_{\text{int}}$ stores intervention state-action pairs $(s_t, a_t)$. Optionally, we can initialize $D_\pi$ and $D_{\text{int}}$ with a few offline demonstrations to warm-start training [33]. The backup critic $Q_{TD}$ is updated on $D_\pi$ by minimizing the standard Bellman regression loss,

$$\mathcal{L}_{TD} = \mathbb{E}_{(s, a, r^{\text{task}}_t, s') \sim D_\pi}\!\left[\Big(Q_{TD}(s, a) - \big(r^{\text{task}}_t + \gamma\, \mathbb{E}_{a' \sim \pi_\theta(\cdot \mid s')} Q_{TD}(s', a')\big)\Big)^2\right].$$

For the optimistic critic $Q_{Opt}$, we add the optimistic reward $r^{\text{bonus}}$ but use TD targets from $Q_{TD}$, while the actor $\pi_\theta$ objective combines (i) RL policy improvement using $Q_{Opt}$ and (ii) maximum likelihood on $D_{\text{int}}$:

$$\mathcal{L}_{Opt} = \mathbb{E}_{(s, a, r^{\text{task}}_t, r^{\text{bonus}}, s') \sim D}\!\left[\Big(Q_{Opt}(s, a) - \big(\underbrace{r^{\text{task}}_t + r^{\text{bonus}}}_{\text{adding optimism}} + \underbrace{\gamma\, \mathbb{E}_{a' \sim \pi_\theta(\cdot \mid s')} Q_{TD}(s', a')}_{\text{using TD-backup}}\big)\Big)^2\right]$$

$$\mathcal{L}_\theta = -\underbrace{\mathbb{E}_{s \sim D_\pi,\, a \sim \pi_\theta(\cdot \mid s)}\big[Q_{Opt}(s, a)\big]}_{\text{optimistic policy improvement}} - \underbrace{\mathbb{E}_{(s, a) \sim D_{\text{int}}}\big[\lambda \log \pi_\theta(a \mid s)\big]}_{\text{BC-regularization}}$$

**BC regularization.** Interventions are suboptimal (Section 4.1) but provide a useful early directional signal compared to random exploration (see value improvement over corrections in Figure 2). We therefore regularize the actor with a BC term on intervention actions in $D_{\text{int}}$ with a fixed $\lambda \approx 0.1$.

---

## 6 Experiments

Our experiments aim to answer whether Q-OIL improves the ability to learn from interventions (1) in simulated robot experiments against the proposed model of human behavior; (2) with real humans intervening simulated robots; and (3) with real human interventions on a robot. Finally, (4) ablations investigate how components such as the decoupled critics and BC-regularization affect performance.

### 6.1 Experimental Setup

**Tasks.** We evaluate our method on real and simulated manipulation tasks using both simulated human models and real participants: (i) *Pen-in-Bowl*: grasp a pen and place it in a bowl; (ii) *Unbutton-Shirt*: open a shirt button, requiring fine-grained deformable-object manipulation; (iii) *Peg-Insertion*: pick and precisely insert a randomly placed peg; and (iv) *Simulated Robomimic*: three simulated pick-and-place/assembly tasks from Mandlekar et al. [42].

**Baselines.** We compare against methods spanning the three categories in Section 2: (i) **HIL** [35], which accelerates off-policy RL with online interventions in the replay buffer; (ii) **HG-DAgger** [8], a DAgger [5] variant that treats interventions as BC targets; (iii) **SIRIUS** [41], which applies weighted BC to interventions and on-policy data; (iv) **PVP** [43], which modifies the critic loss to prefer human actions; (v) **RLIF** [17], which assigns negative reward to pre-intervention states during off-policy RL; and (vi) **BC+RL** [44], which combines BC on intervention actions with off-policy RL.

**Training Details.** We use a JAX-based implementation built on RLPD [33]. In simulation, we use our synthetic model PAM for reliable benchmarking and report success rates over 100k environment interactions. We conduct experiments with four participants intervening across two state-based simulated tasks. We narrow the reset distribution when real humans intervene in simulation to reduce training time. For real-robot tasks, we train RGB policies with wrist-mounted and third-person cameras on the Franka Panda robot and run training with a single human across three tasks. We warm-start experiments with 10–20 demonstrations, and evaluate each checkpoint for 10 episodes (100 for simulated tasks). Details for setup, tasks and hyperparameters are in Appendices E and I.

### 6.2 Results

> **Figure 6: Left:** Evaluation tasks in simulation and on the Franka robot. **Right:** On Robomimic with the progress-aware human model (PAM), Q-OIL improves success rate and learning efficiency over baselines.

**Does Q-OIL improve success rate and learning efficiency in simulated tasks?**

In Figure 6 (right), we evaluate Q-OIL on Robomimic using our progress-aware human model (PAM), which we show (Section 4.1) better aligns with real human interventions and evaluations. Across tasks, Q-OIL consistently achieves higher success rates with fewer interactions than baselines. This highlights that optimistic exploration with noisy corrections is more effective than directly imitating, penalizing, or naively adding interventions to the replay buffer. We also compare all methods under existing human models and ablate expert quality in Appendices F and H.

> **Figure 7:** Q-OIL outperforms baselines when human users interact with simulated robots across two tasks.

**Does Q-OIL improve task performance and reduce real human effort when participants interact with simulated robots?**

To evaluate performance under real human interventions, we recruited four participants to intervene on two Robomimic tasks across six methods, yielding 48 runs; see Appendix D.2 for details. In Figure 7 (top), we observe that Q-OIL outperforms all baselines with real humans, consistent with our simulated intervention results. HG-DAgger [8] has non-trivial performance but over-relies on imitating the interventions (which are suboptimal), and therefore underperforms. RLIF [17], which penalizes states preceding interventions, is overly pessimistic and fails to learn within the interaction budget. In contrast, Q-OIL uses optimism to explore the human-guided regions and rapidly learns a policy with a high success rate. We also find that BC regularization enables faster and lower variance learning in the longer-horizon ($\sim 400$ steps) Can task. We observe similar trends between real-human interventions and our synthetic intervention model, supporting the fidelity of our data-driven evaluation setup. Figure 7 (bottom) shows that Q-OIL requires fewer interventions than the baselines, while HIL [35] requires substantially more intervention steps.

> **Figure 8:** Across three tasks, Q-OIL achieves higher success and throughput given a fixed interaction budget, with significantly lower human effort.

**Does Q-OIL improve efficiency and performance on real robot tasks?**

We validate Q-OIL on real robot hardware using a Franka Panda (Appendix E), across three different tasks and RL baselines. Our analysis in Section 4.1 shows that humans could be noisy and divergent, and Q-OIL robustly utilizes interventions to learn a policy with $10$–$50\%$ higher success rates and almost $30\%$ lower effort (Figure 8). By biasing exploration toward human-corrected regions, Q-OIL collects more task-relevant experience within the same interaction budget and learns policies with an average $90\%$ higher execution efficiency, as reflected in higher throughput, measured as successes per 1k steps. We include complete videos of our training runs, and some qualitative discussion on our website. Finally, in Figure 8, we observe that Q-OIL is the most efficient in terms of interventions across tasks, i.e., it makes maximal use of each human correction to adapt its behavior. Together with our simulation results under synthetic and real corrections, these results show that human-grounded evaluation leads to better algorithmic design that transfers to real users and robots. HIL fails to learn within a smaller budget, so, for a task we run baselines for additional steps to get non-trivial task successes.

> **Figure 9:** Q-OIL w/o decoupling degrades with higher bonus due to over-optimism.

**Are the decoupled critics essential for Q-OIL stability?**

In Figure 9, we ablate a key design choice: removing the decoupling between the task reward and optimism bonus. We present the success rate averaged over Robomimic tasks with our synthetic human model. In Appendix A, we show a toy example in which removing decoupling leads to catastrophic propagation of positive bonuses to non-intervention states, causing over-optimism and higher values for suboptimal states. This severely harms performance, as shown in Figure 9, where performance decreases as we increase the bonus value for optimism.

---

## 7 Conclusion

In this work, we study real-world human interventions, revealing a significant gap between existing assumptions and actual behavior. Analyzing real intervention data, we find that humans intervene based on non-Markovian progress rather than local suboptimality, and that their corrections are suboptimal and diverge from the converged policy. These findings motivate two contributions: (1) a progress-aware intervention model (PAM) that better supports algorithm development and evaluation by matching real-user outcomes, unlike prior synthetic human models, and (2) Q-OIL, an algorithm that uses interventions to guide exploration by adding an optimism bonus at intervention transitions, while decoupling the critics for backups and policy improvement. Across simulated and real robot experiments, Q-OIL improves task success by approximately $10$–$50\%$ with $30\%$ less human effort than baselines, providing a practical path toward scalable human-in-the-loop learning.

**Limitations:** Our human study uses 8 users in simulated tasks across 24 experimental runs; a larger and more diverse participant pool in the real world would further strengthen our findings about interventions. Our real-robot experiments use space-mouse-based teleoperation, but other interfaces such as kinesthetic teaching may lead to different intervention patterns. We evaluate Q-OIL on tabletop manipulation tasks; future work should study robots deployed in unstructured settings, where humans intervene across diverse, long-horizon tasks with richer intervention patterns. Finally, scaling Q-OIL to fine-tune large pretrained models remains an interesting direction for future work.

---

## References

[1] P. Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, M. Y. Galliker, D. Ghosh, L. Groom, K. Hausman, B. Ichter, S. Jakubczak, T. Jones, L. Ke, D. LeBlanc, S. Levine, A. Li-Bell, M. Mothukuri, S. Nair, K. Pertsch, A. Z. Ren, L. X. Shi, L. Smith, J. T. Springenberg, K. Stachowicz, J. Tanner, Q. Vuong, H. Walke, A. Walling, H. Wang, L. Yu, and U. Zhilinsky. π0.5: a vision-language-action model with open-world generalization, 2025. URL https://arxiv.org/abs/2504.16054.

[2] X. Zhou, Y. Xu, G. Tie, Y. Chen, G. Zhang, D. Chu, P. Zhou, and L. Sun. Libero-pro: Towards robust and fair evaluation of vision-language-action models beyond memorization. arXiv preprint arXiv:2510.03827, 2025.

[3] Y. Hu, F. Lin, P. Sheng, C. Wen, J. You, and Y. Gao. Data scaling laws in imitation learning for robotic manipulation. arXiv preprint arXiv:2410.18647, 2024.

[4] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, J. Schulman, J. Hilton, F. Kelton, L. Miller, M. Simens, A. Askell, P. Welinder, P. Christiano, J. Leike, and R. Lowe. Training language models to follow instructions with human feedback, 2022.

[5] S. Ross, G. J. Gordon, and J. A. Bagnell. Reduction of imitation learning and structured prediction to no-regret online learning. In AISTATS, pages 627–635, 2011. URL http://www.aaai.org/ocs/index.php/AISTATS/AISTATS11/paper/view/2156.

[6] P. Yin, T. Westenbroek, S. Bagaria, K. Huang, C.-A. Cheng, A. Kolobov, and A. Gupta. Rapidly adapting policies to the real-world via simulation-guided fine-tuning. In International Conference on Learning Representations (ICLR), 2025.

[7] P. Intelligence. π*0.6: a vla that learns from experience. arXiv preprint arXiv:2511.14759, 2025. URL https://arxiv.org/abs/2511.14759.

[8] K. Michael, C. Sidrane, K. Driggs-Campbell, and M. J. Kochenderfer. Hg-dagger: Interactive imitation learning with human experts. In 2019 International Conference on Robotics and Automation (ICRA), pages 5053–5059. IEEE, 2019. URL https://arxiv.org/abs/1810.02890.

[9] Y. Jiang, C. Wang, R. Zhang, J. Wu, and L. Fei-Fei. Transic: Sim-to-real policy transfer by learning from online correction. In Conference on Robot Learning, 2024.

[10] J. Liang, F. Xia, W. Yu, A. Zeng, M. G. Arenas, M. Attarian, M. Bauza, M. Bennice, A. Bewley, A. Dostmohamed, C. Fu, N. Gileadi, M. Giustina, K. Gopalakrishnan, L. Hasenclever, J. Humplik, J. Hsu, N. J. Joshi, B. Jyenis, C. Kew, S. Kirmani, T.-W. E. Lee, K.-H. Lee, A. H. Michaely, J. Moore, K. Oslund, D. Rao, A. Ren, B. Tabanpour, Q. H. Vuong, A. Wahid, T. Xiao, Y. Xu, V. Zhuang, P. Xu, E. Frey, K. Caluwaerts, T.-Y. Zhang, B. Ichter, J. Tompson, L. Takayama, V. Vanhoucke, I. Shafran, M. Mataric, D. Sadigh, N. M. O. Heess, K. Rao, N. Stewart, J. Tan, and C. Parada. Learning to learn faster from human feedback with language model predictive control. ArXiv, abs/2402.11450, 2024. URL https://api.semanticscholar.org/CorpusID:267751232.

[11] A. Bajcsy, D. P. Losey, M. K. O'Malley, and A. D. Dragan. Learning from physical human corrections, one feature at a time. In Proceedings of the 2018 ACM/IEEE International Conference on Human-Robot Interaction, pages 141–149, 2018.

[12] A. Mandlekar, D. Xu, R. Martín-Martín, Y. Zhu, L. Fei-Fei, and S. Savarese. Human-in-the-loop imitation learning using remote teleoperation. ArXiv, abs/2012.06733, 2020. URL https://api.semanticscholar.org/CorpusID:229158088.

[13] J. Spencer, S. Choudhury, M. Barnes, M. Schmittle, M. Chiang, P. Ramadge, and S. Srinivasa. Expert intervention learning. Autonomous Robots, 46(1):99–113, 2022. doi:10.1007/s10514-021-10006-9. URL https://doi.org/10.1007/s10514-021-10006-9.

[14] M. Liu, M. Zhu, and W. Zhang. Goal-conditioned reinforcement learning: Problems and solutions. arXiv preprint arXiv:2201.08299, 2022.

[15] W. B. Knox and P. Stone. Interactively shaping agents via human reinforcement: the tamer framework. In Proceedings of the Fifth International Conference on Knowledge Capture, K-CAP '09, page 9–16, New York, NY, USA, 2009. Association for Computing Machinery. ISBN 9781605586588. doi:10.1145/1597735.1597738. URL https://doi.org/10.1145/1597735.1597738.

[16] A. Xie, F. Tajwar, A. Sharma, and C. Finn. When to ask for help: Proactive interventions in autonomous reinforcement learning. In Advances in Neural Information Processing Systems, volume 35, 2022.

[17] J. Luo, P. Dong, Y. Zhai, Y. Ma, and S. Levine. Rlif: Interactive imitation learning as reinforcement learning, 2024. URL https://arxiv.org/abs/2311.12996.

[18] Y. Korkmaz and E. Biyik. Mile: Model-based intervention learning. 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 15673–15679, 2025. URL https://api.semanticscholar.org/CorpusID:276444697.

[19] D. Lindner, S. Tschiatschek, K. Hofmann, and A. Krause. Interactively learning preference constraints in linear bandits. ArXiv, abs/2206.05255, 2022. URL https://api.semanticscholar.org/CorpusID:249605747.

[20] J. Spencer, S. Choudhury, A. Venkatraman, B. Ziebart, and J. A. Bagnell. Feedback in imitation learning: The three regimes of covariate shift, 2021. URL https://arxiv.org/abs/2102.02872.

[21] S. Ainsworth, M. Barnes, and S. Srinivasa. Mo' states mo' problems: Emergency stop mechanisms from observation, 2019. URL https://arxiv.org/abs/1912.01649.

[22] A. Jain, M. Zhang, K. Arora, W. Chen, M. Torne, M. Z. Irshad, S. Zakharov, Y. Wang, S. Levine, C. Finn, W.-C. Ma, D. Shah, A. Gupta, and K. Pertsch. Polaris: Scalable real-to-sim evaluations for generalist robot policies, 2025. URL https://arxiv.org/abs/2512.16881.

[23] R. Shachnai, M. Kleiman-Weiner, M. Berke, and J. A. Leonard. When bayesians take over: A computational model of parental intervention. In Proceedings of the Annual Meeting of the Cognitive Science Society, volume 47, 2025. URL https://escholarship.org/uc/item/1q346255.

[24] R. S. Sutton. Learning to predict by the methods of temporal differences. Machine Learning, 3(1):9–44, 1988.

[25] B. Argall, S. Chernova, M. Veloso, and B. Browning. A survey of robot learning from demonstration. Robotics and Autonomous Systems, 57(5):469–483, May 2009.

[26] T. Osa, J. Pajarinen, G. Neumann, J. A. Bagnell, P. Abbeel, and J. Peters. An algorithmic perspective on imitation learning. Found. Trends Robotics, 7(1-2):1–179, 2018. doi:10.1561/2300000053. URL https://doi.org/10.1561/2300000053.

[27] M. Memmel, J. Berg, B. Chen, A. Gupta, and J. Francis. Strap: Robot sub-trajectory retrieval for augmented policy learning. In The Thirteenth International Conference on Learning Representations, 2025.

[28] T. Haarnoja, A. Zhou, K. Hartikainen, P. Abbeel, and E. Todorov. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning for continuous control. In Proceedings of the 35th International Conference on Machine Learning, pages 1861–1871. PMLR, 2018. URL http://proceedings.mlr.press/v80/haarnoja18b.html.

[29] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms. CoRR, abs/1707.06347, 2017. URL http://arxiv.org/abs/1707.06347.

[30] A. Rajeswaran, V. Kumar, A. Gupta, G. Vezzani, J. Schulman, E. Todorov, and S. Levine. Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations. In Proceedings of Robotics: Science and Systems (RSS), 2018.

[31] A. Nair, M. Dalal, A. Gupta, and S. Levine. {AWAC}: Accelerating online reinforcement learning with offline datasets, 2021. URL https://openreview.net/forum?id=OJiM1R3jAtZ.

[32] I. Kostrikov, A. Nair, and S. Levine. Offline reinforcement learning with implicit q-learning. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=68n2s9ZJWF8.

[33] P. J. Ball, L. Smith, I. Kostrikov, and S. Levine. Efficient online reinforcement learning with offline data. In Proceedings of the 40th International Conference on Machine Learning, ICML'23. JMLR.org, 2023.

[34] Y. Lu, J. Fu, G. Tucker, X. Pan, E. Bronstein, B. Roelofs, B. Sapp, B. A. White, A. Faust, S. Whiteson, D. Anguelov, and S. Levine. Imitation is not enough: Robustifying imitation with reinforcement learning for challenging driving scenarios. 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 7553–7560, 2022. URL https://api.semanticscholar.org/CorpusID:254974278.

[35] J. Luo, C. Xu, J. Wu, and S. Levine. Precise and dexterous robotic manipulation via human-in-the-loop reinforcement learning. Science Robotics, 10(105):eads5033, 2025.

[36] Q. Li, Z. Peng, and B. Zhou. Efficient learning of safe driving policy via human-ai copilot optimization. arXiv preprint arXiv:2202.10341, 2022. URL https://arxiv.org/abs/2202.10341.

[37] R. I. Brafman and M. Tennenholtz. R-max - a general polynomial time algorithm for near-optimal reinforcement learning. J. Mach. Learn. Res., 3(null):213–231, Mar. 2003. ISSN 1532-4435. doi:10.1162/153244303765208377. URL https://doi.org/10.1162/153244303765208377.

[38] D. Pathak, P. Agrawal, A. A. Efros, and T. Darrell. Curiosity-driven exploration by self-supervised prediction. In ICML, 2017.

[39] E. Bıyık, D. P. Losey, M. Palan, N. C. Landolfi, G. Shevchuk, and D. Sadigh. Learning reward functions from diverse sources of human feedback: Optimally integrating demonstrations and preferences. The International Journal of Robotics Research, 41(1):45–67, 2022.

[40] D. Kahneman and A. Tversky. Prospect theory: An analysis of decision under risk. Econometrica, 47(2):263–291, 1979. ISSN 00129682, 14680262. URL http://www.jstor.org/stable/1914185.

[41] H. Liu, S. Nasiriany, L. Zhang, Z. Bao, and Y. Zhu. Robot learning on the job: Human-in-the-loop autonomy and learning during deployment. In Robotics: Science and Systems (RSS), 2023.

[42] A. Mandlekar, D. Xu, J. Wong, S. Nasiriany, C. Wang, R. Kulkarni, L. Fei-Fei, S. Savarese, Y. Zhu, and R. Martín-Martín. What matters in learning from offline human demonstrations for robot manipulation. arXiv preprint arXiv:2108.03298, 2021.

[43] Z. Peng, W. Mo, C. Duan, Q. Li, and B. Zhou. Learning from active human involvement through proxy value propagation. In Advances in Neural Information Processing Systems (NeurIPS), 2023. URL https://papers.nips.cc/paper_files/paper/2023/file/f57ffe47d0b528fbb97901d16bd4eba2-Paper-Conference.pdf.

[44] S. Fujimoto and S. S. Gu. A minimalist approach to offline reinforcement learning. arXiv preprint arXiv:2106.06860, 2021. URL https://arxiv.org/abs/2106.06860.
