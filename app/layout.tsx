import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "RRNN | Reinventando las Redes Neuronales",
  description: "Soluciones innovadoras de inteligencia artificial y redes neuronales que redefinen el futuro de los sistemas inteligentes.",
  keywords: ["IA", "Redes Neuronales", "Aprendizaje Automático", "Aprendizaje Profundo", "RRNN", "Inteligencia Artificial"],
  authors: [{ name: "Equipo RRNN", url: "https://rrnn.ai" }],
  openGraph: {
    title: "RRNN | Reinventando las Redes Neuronales",
    description: "Soluciones innovadoras de inteligencia artificial y redes neuronales que redefinen el futuro de los sistemas inteligentes.",
    url: "https://rrnn.ai",
    siteName: "RRNN",
    images: [
      {
        url: "https://rrnn.ai/og-image.jpg",
        width: 1200,
        height: 630,
        alt: "RRNN - Reinventando las Redes Neuronales",
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
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
