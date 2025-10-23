function [E, dpsi] = energy_fd(psi, dx, g)
    % Centered FD for ψ_x and discrete energy:
    % H = ∫ ( 1/2 |ψ_x|^2  - g/2 |ψ|^4 ) dx
    % Periodic stencil
    N   = numel(psi);
    psi_f = psi([2:N 1]);
    psi_b = psi([N 1:N-1]);
    dpsi  = (psi_f - psi_b) / (2*dx);
    term1 = 0.5 * abs(dpsi).^2;
    term2 = -0.5 * g * abs(psi).^4;
    E     = dx * sum(term1 + term2);
end
