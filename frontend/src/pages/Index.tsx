import { useState, useCallback } from "react";
import { toast } from "sonner";
import { Sidebar } from "@/components/Sidebar";
import { Canvas } from "@/components/Canvas";
import { PromptBar } from "@/components/PromptBar";
import { MobileDrawer } from "@/components/MobileDrawer";
import { MobileHeader } from "@/components/MobileHeader";
import { ImageModal } from "@/components/ImageModal";

// Placeholder images for mock generation
const PLACEHOLDER_IMAGES = [
  "https://images.unsplash.com/photo-1620641788421-7a1c342ea42e?w=768&h=768&fit=crop&q=80",
  "https://images.unsplash.com/photo-1614850523296-d8c1af93d400?w=768&h=768&fit=crop&q=80",
  "https://images.unsplash.com/photo-1633356122544-f134324a6cee?w=768&h=768&fit=crop&q=80",
  "https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?w=768&h=768&fit=crop&q=80",
];

const Index = () => {
  // Generation parameters
  const [steps, setSteps] = useState(30);
  const [cfg, setCfg] = useState(7.5);
  const [seed, setSeed] = useState(42);
  const [dimensions, setDimensions] = useState<"512x512" | "768x768">("768x768");

  // Prompt state
  const [prompt, setPrompt] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");

  // UI state
  const [isLoading, setIsLoading] = useState(false);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleGenerate = useCallback(() => {
    if (!prompt.trim()) {
      toast.error("Please enter a prompt");
      return;
    }

    setIsLoading(true);
    setGeneratedImage(null);

    // Simulate API call
    setTimeout(() => {
      const randomImage = PLACEHOLDER_IMAGES[Math.floor(Math.random() * PLACEHOLDER_IMAGES.length)];
      setGeneratedImage(randomImage);
      setIsLoading(false);
      toast.success("Image generated successfully!");
    }, 3000);
  }, [prompt]);

  const handleDownload = useCallback(() => {
    if (!generatedImage) return;

    // Create a link to download the image
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
