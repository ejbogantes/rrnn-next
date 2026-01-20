import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Detecta efectos secundarios y bugs en render/animaciones
  reactStrictMode: true,

  // Reduce ruido y expone menos información
  poweredByHeader: false,

  // Preparado para assets educativos y diagramas
  images: {
    formats: ["image/avif", "image/webp"],
  },

};

export default nextConfig;