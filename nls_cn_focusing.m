function nls_cn_focusing
    % ---------- Problem & numerics ----------
    g    = 1.0;            % focusing strength (>0)
    L    = 40;             % domain length [-L/2, L/2]
    N    = 512;            % grid points (use power of 2 if you also test FFT diagnostics)
    dx   = L/N;
    x    = (-N/2:N/2-1).' * dx;

    dt   = 1e-3;           % time step
    T    = 2;            % final time
    nSteps = round(T/dt);

    % ---------- Initial condition: bright soliton ----------
    % Analytic soliton for iψ_t + ψ_xx + 2|ψ|^2 ψ = 0
    % Our PDE is iψ_t = -(1/2)ψ_xx + g|ψ|^2ψ.
    % Map by scaling: choose parameters so it’s stationary.
    A  = 1.0;  % 1.5            % amplitude parameter
    v  = 0.5;  % 0            % velocity; try e.g. 0.5 for a traveling soliton
    % psi = A * sech(A*(x - 0)) .* exp(1i*(0.5*v*x));  % good test field, bright soliton
    % psi = A * sin(v*x); % sin wave
    % psi = A * exp(1i*v*x); % plane wave
     noise_amp = 0.03;             % relative noise level
     psi = A * (1 + noise_amp*(randn(size(x)) + 1i*randn(size(x)))); % unstable

    % ---------- Precompute discrete Laplacian (periodic) ----------
    e = ones(N,1);
    Lap = spdiags([e -2*e e], [-1 0 1], N, N);
    Lap(1,N) = 1; Lap(N,1) = 1;                % periodic wrap
    % CN has average of Lap^n and Lap^{n+1} with prefactor -(1/4dx^2)
    Lop = -(1/(4*dx^2)) * Lap;                 % this is the CN linear operator piece

    I  = speye(N);
    % ---------- Diagnostics storage ----------
    mass0   = trapz(x, abs(psi).^2);
    [E0, ~] = energy_fd(psi, dx, g);           % discrete energy
    mass_t  = zeros(nSteps,1);
    energy_t= zeros(nSteps,1);
    maxpsi_t= zeros(nSteps,1);

    % ---------- Time stepping ----------
    for n = 1:nSteps
        psi_old = psi;

        % Right-hand side:  i/dt*psi^n  - (1/4dx^2) Lap psi^n  + (g/2)|psi^n|^2 psi^n
        b = (1i/dt)*psi_old + Lop*psi_old + (g/2) * (abs(psi_old).^2 .* psi_old);

    % ---------- Fixed-point / Picard iteration (no toolbox) ----------
    psi = psi_old;  % initial guess
    for it = 1:6    % usually 3–6 iterations are enough
        nonlinear_term = (g/2) * abs(psi).^2 .* psi;
        rhs = b + nonlinear_term;
        psi_new = ((1i/dt)*speye(N) + Lop) \ rhs;
        
        % optional damping (helps stability for strong focusing)
        psi = 0.7*psi_new + 0.3*psi;
        
        % convergence check (can omit for simplicity)
        if norm(psi_new - psi)/norm(psi) < 1e-8
            psi = psi_new;
            break;
        end
    end

        % ---------- Diagnostics ----------
        mass_t(n)    = trapz(x, abs(psi).^2) / mass0 - 1;
        [E, dpsi]    = energy_fd(psi, dx, g);
        energy_t(n)  = (E/E0) - 1;
        maxpsi_t(n)  = max(abs(psi));

        % ---------- Simple live plots ----------
        if mod(n, round(0.02/dt)) == 0 || n==1 || n==nSteps
            subplot(2,2,1), plot(x, abs(psi).^2, 'LineWidth',1); grid on
            xlabel('x'), ylabel('|ψ|^2'), title('Intensity')
            xlim([x(1), x(end)]); ylim([0, 1.2*max(abs(psi).^2)+eps])

            subplot(2,2,2), plot(x, real(psi), x, imag(psi), 'LineWidth',1); grid on
            xlabel('x'), ylabel('Re/Im ψ'), title('Field')

            subplot(2,2,[3 4])
            yyaxis left
            plot((1:n)*dt, mass_t(1:n), 'LineWidth',1); ylabel('Rel mass drift')
            yyaxis right
            plot((1:n)*dt, energy_t(1:n), 'LineWidth',1); ylabel('Rel energy drift')
            xlabel('t'), title(sprintf('Conservation; max|ψ|=%.3g', maxpsi_t(n))); grid on
            drawnow
        end
    end

    fprintf('Final rel mass drift:  %.3e\n', mass_t(end));
    fprintf('Final rel energy drift: %.3e\n', energy_t(end));
end

