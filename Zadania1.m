
%% Zadanie I.1
clear all;
close all;
clc;

grubosc = 1.25; set(groot,'DefaultLineLineWidth',grubosc); set(groot,'DefaultStairLineWidth',grubosc);
colors = lines; set(groot, 'defaultAxesColorOrder', colors);

u_min = -1;
u_max =  1;
Nu = 100;
u_vals = linspace(u_min,u_max,Nu);

k_min = 10;
k_max = 70;

y_stat = zeros(1,Nu);

for i = 1:Nu
    u = u_vals(i) * ones(1,k_max);
    x = zeros(1,k_max);
    y = zeros(1,k_max);

    for k = k_min:k_max
        [y(k), x(k)] = proces12_symulator(u(k-5), u(k-6), x(k-1), x(k-2));
    end

    y_stat(i) = y(k_max);
end

figure;
plot(u_vals, y_stat);
xlabel('u');
ylabel('y');
title('Charakterystyka statyczna procesu');
grid on;

%% Zadanie I.2

%----------Dane uczące----------
rng(42)

N = 2200;
T_change = 30;      

nBlocks = ceil(N/T_change);
u_levels = u_min + (u_max-u_min)*rand(1, nBlocks);
u_ucz = repelem(u_levels, T_change);
u_ucz = u_ucz(1:N);

y_ucz = zeros(1,N);
x = zeros(1,N);

for k = 7:N
    [y_ucz(k), x(k)] = proces12_symulator( ...
        u_ucz(k-5), u_ucz(k-6), x(k-1), x(k-2));
end


%--------Dane weryfikujace--------

rng(41)

nBlocks = ceil(N/T_change);
u_levels = u_min + (u_max-u_min)*rand(1, nBlocks);
u_wer = repelem(u_levels, T_change);
u_wer = u_wer(1:N);

y_wer = zeros(1,N);
x = zeros(1,N);

for k = 7:N
    [y_wer(k), x(k)] = proces12_symulator( ...
        u_wer(k-5), u_wer(k-6), x(k-1), x(k-2));
end


k = 1:length(u_wer);

figure;
subplot(2,1,1);
stairs(k, u_ucz);
ylabel('u(k)');
title('Dane uczące');
grid on;

subplot(2,1,2);
plot(k, y_ucz);
ylabel('y(k)');
xlabel('k');
grid on;

figure;
subplot(2,1,1);
stairs(k, u_wer);
ylabel('u(k)');
title('Dane weryfikujące');
grid on;

subplot(2,1,2);
plot(k, y_wer);
ylabel('y(k)');
xlabel('k');
grid on;

%% Sekcja II – ELM (K = 5 neuronów ukrytych)
close all
clc;

%% --- parametry ---
K = 20;          % liczba neuronów ukrytych
k0 = 7;         % pierwszy poprawny indeks (wynika z opóźnień)

N_ucz = length(u_ucz);
N_wer = length(u_wer);

%% --- budowa danych uczących ---
Nu = N_ucz - k0 + 1;
X_ucz = zeros(4, Nu);
Y_ucz = zeros(1, Nu);

idx = 1;
for k = k0:N_ucz
    X_ucz(:,idx) = [u_ucz(k-5),u_ucz(k-6),y_ucz(k-1),y_ucz(k-2)];
    Y_ucz(idx) = y_ucz(k);
    idx = idx + 1;
end

%% --- budowa danych weryfikujących ---
Nv = N_wer - k0 + 1;
X_wer = zeros(4, Nv);
Y_wer = zeros(1, Nv);

idx = 1;
for k = k0:N_wer
    X_wer(:,idx) = [ ...
        u_wer(k-5);
        u_wer(k-6);
        y_wer(k-1);
        y_wer(k-2)];
    Y_wer(idx) = y_wer(k);
    idx = idx + 1;
end

%% --- uczenie ELM ---
inputSize = 4;

% losowe wagi pierwszej warstwy
w10 = randn(K,1);            % biasy
w1  = randn(K,inputSize);    % wagi wejściowe

% wyjścia warstwy ukrytej
H_ucz = tanh(w10 + w1 * X_ucz);

% uczenie wag wyjściowych (LS)
W =(H_ucz.' \ Y_ucz.').';

%% tryb bez rekurencji
Yhat_ucz = W * tanh(w10 + w1 * X_ucz);
Yhat_wer = W * tanh(w10 + w1 * X_wer);

MSE_ucz = mean((Y_ucz - Yhat_ucz).^2);
MSE_wer = mean((Y_wer - Yhat_wer).^2);

fprintf('ELM K=5 | one-step | MSE ucz = %.3e | MSE wer = %.3e\n', ...
        MSE_ucz, MSE_wer);

%% --- tryb z rekurencją (free-run) ---

% === ZBIÓR UCZĄCY ===
y_hat_ucz = zeros(1, N_ucz);

% inicjalizacja (prawdziwe wyjścia tylko na start)
y_hat_ucz(1:2) = y_ucz(1:2);

for k = k0:N_ucz
    xk = [ ...
        u_ucz(k-5);
        u_ucz(k-6);
        y_hat_ucz(k-1);
        y_hat_ucz(k-2)];
    y_hat_ucz(k) = W * tanh(w10 + w1 * xk);
end

MSE_rec_ucz = mean((y_ucz(k0:N_ucz) - y_hat_ucz(k0:N_ucz)).^2);

% === ZBIÓR WERYFIKUJĄCY ===
y_hat_wer = zeros(1, N_wer);

% inicjalizacja (prawdziwe wyjścia tylko na start)
y_hat_wer(1:2) = y_wer(1:2);

for k = k0:N_wer
    xk = [ ...
        u_wer(k-5);
        u_wer(k-6);
        y_hat_wer(k-1);
        y_hat_wer(k-2)];
    y_hat_wer(k) = W * tanh(w10 + w1 * xk);
end

MSE_rec_wer = mean((y_wer(k0:N_wer) - y_hat_wer(k0:N_wer)).^2);

fprintf('ELM K=5 | rekurencja | MSE ucz = %.3e | MSE wer = %.3e\n', ...
        MSE_rec_ucz, MSE_rec_wer);


%% --- wykres (rekurencja, weryfikacja) ---
figure;
plot(k0:N_wer, y_wer(k0:N_wer), 'b');
hold on;
plot(k0:N_wer, y_hat_wer(k0:N_wer), 'r--');
grid on;
legend('Proces','Model ELM');
xlabel('k');
ylabel('y');
title('ELM K = 5 – tryb rekurencyjny (weryfikacja)');

%% III. Modelowanie neuronowe – Neural Network Toolbox
%% Jedna warstwa ukryta, K = 1 neuron, trainlm
%% Model dynamiczny (NARX), proces12
%% WYMAGA: u_ucz, y_ucz, u_wer, y_wer

clearvars -except u_ucz y_ucz u_wer y_wer
close all;
clc;

%% -------------------------------------------------------
%  Parametry wynikające z symulatora
%% -------------------------------------------------------
k0 = 7;        % pierwszy indeks z pełną historią
nIn = 4;       % [u(k-5), u(k-6), y(k-1), y(k-2)]

%% -------------------------------------------------------
%  Budowa danych UCZĄCYCH (one-step-ahead)
%% -------------------------------------------------------
N_ucz = length(u_ucz);
Nu = N_ucz - k0 + 1;

X_ucz = zeros(nIn, Nu);
Y_ucz = zeros(1, Nu);

idx = 1;
for k = k0:N_ucz
    X_ucz(:,idx) = [ ...
        u_ucz(k-5);
        u_ucz(k-6);
        y_ucz(k-1);
        y_ucz(k-2)];
    Y_ucz(idx) = y_ucz(k);
    idx = idx + 1;
end

%% -------------------------------------------------------
%  Budowa danych WERYFIKUJĄCYCH
%% -------------------------------------------------------
N_wer = length(u_wer);
Nv = N_wer - k0 + 1;

X_wer = zeros(nIn, Nv);
Y_wer = zeros(1, Nv);

idx = 1;
for k = k0:N_wer
    X_wer(:,idx) = [ ...
        u_wer(k-5);
        u_wer(k-6);
        y_wer(k-1);
        y_wer(k-2)];
    Y_wer(idx) = y_wer(k);
    idx = idx + 1;
end

%% -------------------------------------------------------
%  Definicja sieci neuronowej
%% -------------------------------------------------------
K = 10;                     % JEDEN neuron ukryty
algorithm = 'trainlm';     % Levenberg–Marquardt

net = feedforwardnet(K, algorithm);

% funkcje aktywacji
net.layers{1}.transferFcn = 'tansig';   % warstwa ukryta
net.layers{2}.transferFcn = 'purelin';  % wyjście

% --- WYMAGANIA Z POLECENIA ---
net.divideFcn = 'dividetrain';     % brak automatycznego podziału
net.inputs{1}.processFcns  = {};   % brak skalowania wejść
net.outputs{1}.processFcns = {};   % brak skalowania wyjścia

% parametry uczenia
net.trainParam.epochs = 300;
net.trainParam.showWindow = false;
net.trainParam.showCommandLine = true;

%% -------------------------------------------------------
%  UCZENIE SIECI
%% -------------------------------------------------------
net = train(net, X_ucz, Y_ucz);

%% -------------------------------------------------------
%  TRYB BEZ REKURENCJI (one-step-ahead)
%% -------------------------------------------------------
Yhat_ucz_1s = net(X_ucz);
Yhat_wer_1s = net(X_wer);

MSE_ucz_1s = mean((Y_ucz - Yhat_ucz_1s).^2);
MSE_wer_1s = mean((Y_wer - Yhat_wer_1s).^2);

fprintf('NN K=1 | one-step | MSE ucz = %.3e | MSE wer = %.3e\n', ...
        MSE_ucz_1s, MSE_wer_1s);

%% -------------------------------------------------------
%  TRYB Z REKURENCJĄ (free-run)
%% -------------------------------------------------------

% === ZBIÓR UCZĄCY ===
y_hat_ucz = zeros(1, N_ucz);

% inicjalizacja pamięci (prawdziwe wyjścia tylko na start)
y_hat_ucz(1:2) = y_ucz(1:2);

for k = k0:N_ucz
    xk = [ ...
        u_ucz(k-5);
        u_ucz(k-6);
        y_hat_ucz(k-1);
        y_hat_ucz(k-2)];
    y_hat_ucz(k) = net(xk);
end

MSE_ucz_rec = mean((y_ucz(k0:N_ucz) - y_hat_ucz(k0:N_ucz)).^2);

% === ZBIÓR WERYFIKUJĄCY ===
y_hat_wer = zeros(1, N_wer);

% inicjalizacja pamięci (prawdziwe wyjścia tylko na start)
y_hat_wer(1:2) = y_wer(1:2);

for k = k0:N_wer
    xk = [ ...
        u_wer(k-5);
        u_wer(k-6);
        y_hat_wer(k-1);
        y_hat_wer(k-2)];
    y_hat_wer(k) = net(xk);
end

MSE_wer_rec = mean((y_wer(k0:N_wer) - y_hat_wer(k0:N_wer)).^2);

fprintf('NN K=1 | rekurencja | MSE ucz = %.3e | MSE wer = %.3e\n', ...
        MSE_ucz_rec, MSE_wer_rec);

%% -------------------------------------------------------
%  WYKRES – tryb rekurencyjny (weryfikacja)
%% -------------------------------------------------------
figure;
plot(k0:N_wer, y_wer(k0:N_wer), 'b', 'LineWidth', 1.2);
hold on;
plot(k0:N_wer, y_hat_wer(k0:N_wer), 'r--', 'LineWidth', 1.2);
grid on;
legend('Proces','Model NN');
xlabel('k');
ylabel('y');
title('Neural Network Toolbox – K = 1, tryb rekurencyjny');



