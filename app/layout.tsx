import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

// Fonts optimizadas por Next: evita FOIT y mejora LCP con display: "swap"
const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
  display: "swap",
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  // SEO más alineado con el objetivo educativo/visual del proyecto
  title: "RRNN | Visualizando cómo aprenden las redes neuronales",
  description:
    "Explora y entiende cómo funcionan las redes neuronales mediante visualizaciones interactivas, entrenamiento paso a paso y simulaciones educativas.",
  keywords: [
    "redes neuronales",
    "visualización",
    "aprendizaje automático",
    "machine learning",
    "deep learning",
    "educación en IA",
    "IA generativa",
    "simulación de redes neuronales",
  ],
  // Autor real (mejor para credibilidad en un proyecto educativo)
  authors: [{ name: "Emilio Bogantes", url: "https://rrnn.ai" }],

  // Open Graph optimizado para cuando compartas el proyecto (clase, redes, repo)
  openGraph: {
    title: "RRNN | Aprende cómo funcionan las redes neuronales",
    description:
      "Visualizaciones interactivas que muestran paso a paso cómo una red neuronal aprende, ajusta pesos y reduce el error.",
    url: "https://rrnn.ai",
    siteName: "RRNN",
    images: [
      {
        url: "https://rrnn.ai/og-image.jpg",
        width: 1200,
        height: 630,
        alt: "RRNN - Visualización educativa de redes neuronales",
      },
    ],
    locale: "es_CR",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    // Idioma consistente con el contenido (A11y + SEO)
    <html lang="es-CR">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        {/* Landmark semántico: mejora navegación con lector de pantalla */}
        <main>{children}</main>
      </body>
    </html>
  );
}