%% III. Modelowanie neuronowe – Neural Network Toolbox
%Jedna warstwa ukryta, K = 1 neuron, trainlm
%Model dynamiczny (NARX), proces12
%WYMAGA: u_ucz, y_ucz, u_wer, y_wer

close all;
clc;

%  Parametry wynikające z symulatora
k0 = 7;        % pierwszy indeks z pełną historią
nIn = 4;       % [u(k-5), u(k-6), y(k-1), y(k-2)]


%  Budowa danych UCZĄCYCH (one-step-ahead)

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

%  Budowa danych WERYFIKUJĄCYCH

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

%  Definicja sieci neuronowej

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


%  UCZENIE SIECI

net = train(net, X_ucz, Y_ucz);


%  TRYB BEZ REKURENCJI (one-step-ahead)

Yhat_ucz_1s = net(X_ucz);
Yhat_wer_1s = net(X_wer);

MSE_ucz_1s = mean((Y_ucz - Yhat_ucz_1s).^2);
MSE_wer_1s = mean((Y_wer - Yhat_wer_1s).^2);

fprintf('NN K=1 | one-step | MSE ucz = %.3e | MSE wer = %.3e\n', ...
        MSE_ucz_1s, MSE_wer_1s);


%  TRYB Z REKURENCJĄ (free-run)


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

%  WYKRES – tryb rekurencyjny (weryfikacja)

figure;
plot(k0:N_wer, y_wer(k0:N_wer), 'b', 'LineWidth', 1.2);
hold on;
plot(k0:N_wer, y_hat_wer(k0:N_wer), 'r--', 'LineWidth', 1.2);
grid on;
legend('Proces','Model NN');
xlabel('k');
ylabel('y');
title('Neural Network Toolbox – K = 1, tryb rekurencyjny');