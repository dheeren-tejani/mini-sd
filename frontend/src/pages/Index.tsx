// frontend/src/pages/Index.tsx

import { useState, useCallback } from "react";
import { toast } from "sonner";
import { Sidebar } from "@/components/Sidebar";
import { Canvas } from "@/components/Canvas";
import { PromptBar } from "@/components/PromptBar";
import { MobileDrawer } from "@/components/MobileDrawer";
import { MobileHeader } from "@/components/MobileHeader";
import { ImageModal } from "@/components/ImageModal";

const Index = () => {
  // Generation parameters
  const [steps, setSteps] = useState(30);
  const [cfg, setCfg] = useState(7.5);
  const [seed, setSeed] = useState(42);
  const [dimensions, setDimensions] = useState<"512x512" | "768x768">("512x512");

  // Prompt state
  const [prompt, setPrompt] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");

  // UI state
  const [isLoading, setIsLoading] = useState(false);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);

  // ---------------------------------------------------------
  // ⚡️ REAL BACKEND CONNECTION
  // ---------------------------------------------------------
  const handleGenerate = useCallback(async () => {
    if (!prompt.trim()) {
      toast.error("Please enter a prompt");
      return;
    }

    setIsLoading(true);
    setGeneratedImage(null);

    try {
      // 1. Send Request to your FastAPI Backend
      const response = await fetch("http://localhost:8000/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt: prompt,
          steps: steps,
          cfg_scale: cfg,    // Maps 'cfg' to 'cfg_scale' for backend
          seed: seed,
          // Note: Backend currently ignores dimensions/negative_prompt 
          // but we send them for future-proofing
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Generation failed");
      }

      // 2. Get the Base64 Image
      const data = await response.json();
      
      // 3. Display it
      setGeneratedImage(data.image); 
      toast.success("Image generated successfully!");

    } catch (error) {
      console.error("Generation Error:", error);
      toast.error("Failed to connect. Is the backend running?");
    } finally {
      setIsLoading(false);
    }
  }, [prompt, steps, cfg, seed]); // Dependencies

  // ... (Keep handleDownload and handleExpand exactly the same)
  const handleDownload = useCallback(() => {
    if (!generatedImage) return;
    const link = document.createElement("a");
    link.href = generatedImage;
    link.download = `rangeflow-${Date.now()}.png`;
    link.target = "_blank";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    toast.success("Download started!");
  }, [generatedImage]);

  const handleExpand = useCallback(() => {
    setIsModalOpen(true);
  }, []);

  return (
    <div className="flex h-screen flex-col bg-background">
      {/* Mobile Header */}
      <MobileHeader onOpenDrawer={() => setIsDrawerOpen(true)} />

      {/* Mobile Drawer */}
      <MobileDrawer
        isOpen={isDrawerOpen}
        onClose={() => setIsDrawerOpen(false)}
        steps={steps}
        setSteps={setSteps}
        cfg={cfg}
        setCfg={setCfg}
        seed={seed}
        setSeed={setSeed}
        dimensions={dimensions}
        setDimensions={setDimensions}
        isReady={!isLoading}
      />

      <div className="flex flex-1 overflow-hidden">
        {/* Desktop Sidebar */}
        <Sidebar
          steps={steps}
          setSteps={setSteps}
          cfg={cfg}
          setCfg={setCfg}
          seed={seed}
          setSeed={setSeed}
          dimensions={dimensions}
          setDimensions={setDimensions}
          isReady={!isLoading}
        />

        {/* Main Content */}
        <main className="flex flex-1 flex-col overflow-hidden">
          {/* Canvas */}
          <Canvas
            image={generatedImage}
            isLoading={isLoading}
            onDownload={handleDownload}
            onExpand={handleExpand}
          />

          {/* Prompt Bar */}
          <PromptBar
            prompt={prompt}
            setPrompt={setPrompt}
            negativePrompt={negativePrompt}
            setNegativePrompt={setNegativePrompt}
            onGenerate={handleGenerate}
            isLoading={isLoading}
          />
        </main>
      </div>

      {/* Fullscreen Image Modal */}
      <ImageModal
        image={generatedImage}
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onDownload={handleDownload}
      />
    </div>
  );
};

export default Index;