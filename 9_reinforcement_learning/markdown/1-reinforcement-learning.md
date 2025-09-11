---
time: 03/17/2023
title: Học tăng cường Reinforcement Learning
description: Reinforcement Learning là một mô hình học máy trong đó một hệ thống học tập tự động từ kinh nghiệm tương tác với môi trường. Nó liên quan đến việc ra quyết định nào đó để tối đa hóa một phần thưởng tích lũy. Reinforcement Learning mô phỏng quá trình học tập của con người và động vật, trong đó họ học từ kinh nghiệm và phản hồi từ môi trường.
banner_url: 
tags: [deep-learning, reinforcement-learning]
is_highlight: false
is_published: false
---

## 1. Giới thiệu chung về Reinforcement learning

Reinforcement Learning là một mô hình học máy trong đó một hệ thống học tập tự động từ kinh nghiệm tương tác với môi trường.
Nó liên quan đến việc ra quyết định nào đó để tối đa hóa một phần thưởng tích lũy.

### 1.1. Các ứng dụng của Reinforcement learning

RL có nhiều ứng dụng quan trọng:
Game AI: AlphaGo (chơi cờ vây) dùng RL và mạng học sâu để đánh bại cao thủ cờ vây
stable-baselines3.readthedocs.io
. OpenAI Five huấn luyện đối kháng trong Dota 2 đạt trình độ chuyên nghiệp.
Điều khiển và Robotics: Robot học cách di chuyển, giữ thăng bằng, điều khiển cánh tay robot để thực hiện nhiệm vụ nhặt và đặt đồ. Ví dụ, robot ANYmal được huấn luyện trong mô phỏng với RL rồi áp dụng ra thực tế
stable-baselines3.readthedocs.io
.
Tài chính và tối ưu hóa: RL được dùng để thiết kế chiến lược giao dịch chứng khoán, quản lý danh mục đầu tư, tối ưu hóa dịch vụ (như đặt giá động).
Kỹ thuật lưu lượng và mạng: tối ưu hóa dòng chảy giao thông (đèn tín hiệu tự động), quản lý năng lượng lưới điện.
Khuyến nghị và tương tác: Các hệ thống khuyến nghị nội dung (ví dụ YouTube, Netflix) dùng RL để tối đa hóa sự tương tác lâu dài của người dùng bằng cách cân nhắc ảnh hưởng dài hạn của khuyến nghị.

### 1.2. So sánh Reinforcement learning với các phương pháp học máy khác

Cách chuẩn bị dữ liệu và cách huấn luyện mô hình giữa Reinforcement learning và Supervised learning khác nhau.

Supervised learning

- Dữ liệu huấn luyện của Supervised learning bao gồm cặp dữ liệu đầu vào và đầu ra.
- Mục tiêu của Supervised learning là tìm ra một hàm số ánh xạ từ dữ liệu đầu vào sang dữ liệu đầu ra.

Reinforcement learning

- Dữ liệu huấn luyện của Reinforcement learning bao gồm cặp dữ liệu đầu vào, hành động và phần thưởng.
- Mục tiêu của Reinforcement learning là tìm ra một Policy tối ưu để tối đa hóa phần thưởng.

### 1.3. Ví dụ về bộ dữ liệu trong Reinforcement learning

## 2. Các khái niệm cơ bản trong Reinforcement learning

### 3.1. Ví dụ về game Pacman

- Agent là nhân vật Pacman.
    - Action của Pacman là di chuyển lên, di chuyển xuống, di chuyển trái, di chuyển phải.
    - Policy của Pacman là mô hình machine learning mà chúng ta cần huấn luyện.
- Environment là bản đồ của game.
    - State của môi trường là vị trí của Pacman, các quả bóng, các con ma.
    - Model của môi trường là cách mà các con ma di chuyển, cách mà các quả bóng xuất hiện.
    - Reward của môi trường là điểm số mà Pacman nhận được.
    - Value function của môi trường là cách tính điểm số của Pacman.

### 3.2. Ví dụ về Robot lau nhà

- Agent là robot lau nhà.
    - Action của robot là di chuyển lên, di chuyển xuống, di chuyển trái, di chuyển phải, hút bụi, đổ nước, lau nhà, giặt giẻ lau nhà ...
    - Policy của robot là mô hình machine learning mà chúng ta cần huấn luyện.
- Environment là căn nhà cần lau.
    - State của môi trường là vị trí của robot, vị trí của bụi bẩn.
    - Model của môi trường là cách mà bụi xuất hiện.
    - Reward của môi trường là điểm số mà robot nhận được.
    - Value function của môi trường là cách tính điểm số của robot.

### 2.1. Agent (Tác nhân)

Trong mô hình Reinforcement learning, Agent (Tác nhân) là một thực thể tương tác với môi trường.
Agent có thể thực hiện các Action (Hành động) và nhận được Reward (Phần thưởng) từ môi trường.

Action (Hành động) là một hành động cụ thể mà Agent có thể thực hiện trong một State (Trạng thái) cụ thể.

Policy (Chính sách) là một hàm nhận đầu vào là State (Trạng thái) và trả về Action (Hành động) mà Agent sẽ thực hiện.

$$
action = \pi(state)
$$

trong đó:
- $state$ là trạng thái hiện tại của môi trường.
- $\pi$ là Policy (Chính sách).
- $action$ là hành động mà Agent thực hiện.

Mục tiêu chính của quá trình huấn luyện mô hình Reinforcement learning là tìm ra Policy tối ưu nhất để Agent có thể tối đa hóa Reward (Phần thưởng) mà nó nhận được.

<img src="https://www.kdnuggets.com/wp-content/uploads/awan_reinforcement_learning_newbies_1.png" style="width: 1200px;"/>

### 2.2. Environment (Môi trường)

Environment (Môi trường) là nơi mà Agent tương tác và học hỏi.

### 2.2.1. State (Trạng thái) và Model (Mô hình)

State (Trạng thái) là một tình huống cụ thể mà Agent có thể gặp phải trong môi trường.
Model (Mô hình) là một hàm số nhận đầu vào là State và Action và trả về State tiếp theo.

$$
state' = M(state, action)
$$

trong đó:
- $state$ là trạng thái hiện tại của môi trường.
- $M$ là Model (Mô hình).
- $state'$ là trạng thái tiếp theo của môi trường.

### 2.2.2. Reward (Phần thưởng) và Value function (Hàm giá trị)

Reward (Phần thưởng) là một giá trị mà Agent nhận được từ Environment sau khi thực hiện một hành động.

Value function (Hàm giá trị) là một hàm nhận đầu vào là State (Trạng thái) và trả về Reward (Phần thưởng) mà Agent có thể nhận được từ trạng thái đó.

$$
reward = V(state)
$$

trong đó:
- $state$ là trạng thái hiện tại của môi trường.
- $V$ là Value function (Hàm giá trị).
- $reward$ là phần thưởng mà Agent nhận được.

## 3. Quá trình quyết định Markov (Markov Decision Process - MDP)

## 4. Phương pháp Monte Carlo

## 5. Phân nhóm các thuật toán Reinforcement learning
