%% IV. Modelowanie liniowe procesu (LS)

close all;
clc;

% --- parametry ---
k0 = 7;

%  A) Zbuduj macierze regresji (uczące)

N_ucz = length(u_ucz);
Nu = N_ucz - k0 + 1;

Phi_ucz = zeros(Nu, 5);   % [1, u(k-5), u(k-6), y(k-1), y(k-2)]
Y_ucz_ls = zeros(Nu, 1);  % y(k)

idx = 1;
for k = k0:N_ucz
    Phi_ucz(idx,:) = [1, u_ucz(k-5), u_ucz(k-6), y_ucz(k-1), y_ucz(k-2)];
    Y_ucz_ls(idx) = y_ucz(k);
    idx = idx + 1;
end

%  B) Dopasuj parametry LS

theta = Phi_ucz \ Y_ucz_ls;   % (LS) bardziej numerycznie OK niż inv(...)
% theta = pinv(Phi_ucz)*Y_ucz_ls;  % alternatywa

b0 = theta(1);
b  = theta(2:5);  % [b1 b2 b3 b4]^T

fprintf('Model LS: y = b0 + b1*u(k-5)+b2*u(k-6)+b3*y(k-1)+b4*y(k-2)\n');
fprintf('b0=%.6g, b1=%.6g, b2=%.6g, b3=%.6g, b4=%.6g\n', b0, b(1), b(2), b(3), b(4));


%  C) Tryb bez rekurencji (one-step) – UCZĄCE

Yhat_ucz_1s = Phi_ucz * theta;  % bo w Phi_ucz są prawdziwe y(k-1), y(k-2)
MSE_ucz_1s = mean((Y_ucz_ls - Yhat_ucz_1s).^2);

%  D) Zbuduj macierze regresji (weryfikujące) i one-step

N_wer = length(u_wer);
Nv = N_wer - k0 + 1;

Phi_wer = zeros(Nv, 5);
Y_wer_ls = zeros(Nv, 1);

idx = 1;
for k = k0:N_wer
    Phi_wer(idx,:) = [1, u_wer(k-5), u_wer(k-6), y_wer(k-1), y_wer(k-2)];
    Y_wer_ls(idx) = y_wer(k);
    idx = idx + 1;
end

Yhat_wer_1s = Phi_wer * theta;
MSE_wer_1s = mean((Y_wer_ls - Yhat_wer_1s).^2);


%  E) Tryb z rekurencją (free-run) – UCZĄCE

yhat_ucz_rec = zeros(1, N_ucz);
yhat_ucz_rec(1:2) = y_ucz(1:2); % start z prawdziwych

for k = k0:N_ucz
    xk = [u_ucz(k-5); u_ucz(k-6); yhat_ucz_rec(k-1); yhat_ucz_rec(k-2)];
    yhat_ucz_rec(k) = b0 + b.' * xk;
end

MSE_ucz_rec = mean((y_ucz(k0:N_ucz) - yhat_ucz_rec(k0:N_ucz)).^2);


%  F) Tryb z rekurencją (free-run) – WERYFIKUJĄCE

yhat_wer_rec = zeros(1, N_wer);
yhat_wer_rec(1:2) = y_wer(1:2); % start z prawdziwych

for k = k0:N_wer
    xk = [u_wer(k-5); u_wer(k-6); yhat_wer_rec(k-1); yhat_wer_rec(k-2)];
    yhat_wer_rec(k) = b0 + b.' * xk;
end

MSE_wer_rec = mean((y_wer(k0:N_wer) - yhat_wer_rec(k0:N_wer)).^2);


%  G) Wypisz błędy (4 przypadki)

fprintf('\nLS | one-step  | MSE ucz = %.3e | MSE wer = %.3e\n', MSE_ucz_1s, MSE_wer_1s);
fprintf('LS | rekurencja| MSE ucz = %.3e | MSE wer = %.3e\n', MSE_ucz_rec, MSE_wer_rec);


%  H) Wykresy: oba zbiory, oba tryby


% 1) UCZĄCE – one-step
figure;
plot(k0:N_ucz, y_ucz(k0:N_ucz), 'b', 'LineWidth', 1.2); hold on;
plot(k0:N_ucz, Yhat_ucz_1s,      'r--', 'LineWidth', 1.2);
grid on; legend('Proces','Model LS');
xlabel('k'); ylabel('y');
title('Model LS – UCZĄCE – tryb bez rekurencji (one-step)');

% 2) UCZĄCE – rekurencja
figure;
plot(k0:N_ucz, y_ucz(k0:N_ucz), 'b', 'LineWidth', 1.2); hold on;
plot(k0:N_ucz, yhat_ucz_rec(k0:N_ucz), 'r--', 'LineWidth', 1.2);
grid on; legend('Proces','Model LS');
xlabel('k'); ylabel('y');
title('Model LS – UCZĄCE – tryb rekurencyjny');

% 3) WERYFIKUJĄCE – one-step
figure;
plot(k0:N_wer, y_wer(k0:N_wer), 'b', 'LineWidth', 1.2); hold on;
plot(k0:N_wer, Yhat_wer_1s,     'r--', 'LineWidth', 1.2);
grid on; legend('Proces','Model LS');
xlabel('k'); ylabel('y');
title('Model LS – WERYFIKUJĄCE – tryb bez rekurencji (one-step)');

% 4) WERYFIKUJĄCE – rekurencja
figure;
plot(k0:N_wer, y_wer(k0:N_wer), 'b', 'LineWidth', 1.2); hold on;
plot(k0:N_wer, yhat_wer_rec(k0:N_wer), 'r--', 'LineWidth', 1.2);
grid on; legend('Proces','Model LS');
xlabel('k'); ylabel('y');
title('Model LS – WERYFIKUJĄCE – tryb rekurencyjny');