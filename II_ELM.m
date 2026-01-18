%% II. ELM
close all
clc;

% --- parametry ---
K = 20;          % liczba neuronów ukrytych
k0 = 7;         % pierwszy poprawny indeks (wynika z opóźnień)

N_ucz = length(u_ucz);
N_wer = length(u_wer);

% --- budowa danych uczących ---
Nu = N_ucz - k0 + 1;
X_ucz = zeros(4, Nu);
Y_ucz = zeros(1, Nu);

idx = 1;
for k = k0:N_ucz
    X_ucz(:,idx) = [u_ucz(k-5),u_ucz(k-6),y_ucz(k-1),y_ucz(k-2)];
    Y_ucz(idx) = y_ucz(k);
    idx = idx + 1;
end

% --- budowa danych weryfikujących ---
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

% --- uczenie ELM ---
inputSize = 4;

% losowe wagi pierwszej warstwy
w10 = randn(K,1);            % biasy
w1  = randn(K,inputSize);    % wagi wejściowe

% wyjścia warstwy ukrytej
H_ucz = tanh(w10 + w1 * X_ucz);

% uczenie wag wyjściowych (LS)
W =(H_ucz.' \ Y_ucz.').';

% tryb bez rekurencji
Yhat_ucz = W * tanh(w10 + w1 * X_ucz);
Yhat_wer = W * tanh(w10 + w1 * X_wer);

MSE_ucz = mean((Y_ucz - Yhat_ucz).^2);
MSE_wer = mean((Y_wer - Yhat_wer).^2);

fprintf('ELM K=5 | one-step | MSE ucz = %.3e | MSE wer = %.3e\n', ...
        MSE_ucz, MSE_wer);

% --- tryb z rekurencją (free-run) ---

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


% --- wykres (rekurencja, weryfikacja) ---
figure;
plot(k0:N_wer, y_wer(k0:N_wer), 'b');
hold on;
plot(k0:N_wer, y_hat_wer(k0:N_wer), 'r--');
grid on;
legend('Proces','Model ELM');
xlabel('k');
ylabel('y');
title('ELM K = 5 – tryb rekurencyjny (weryfikacja)');