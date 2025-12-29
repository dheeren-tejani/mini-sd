import { motion, AnimatePresence } from "framer-motion";
import { ImageIcon, Download, Expand, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";

interface CanvasProps {
  image: string | null;
  isLoading: boolean;
  onDownload: () => void;
  onExpand: () => void;
}

const EmptyState = () => (
  <motion.div
    initial={{ opacity: 0, y: 10 }}
    animate={{ opacity: 1, y: 0 }}
    exit={{ opacity: 0, y: -10 }}
    className="flex flex-col items-center justify-center gap-4 text-center"
  >
    <div className="relative">
      <div className="absolute inset-0 rounded-2xl bg-primary/10 blur-2xl" />
      <div className="relative flex h-20 w-20 items-center justify-center rounded-2xl border border-dashed border-border bg-surface-elevated">
        <Sparkles className="h-8 w-8 text-muted-foreground" />
      </div>
    </div>
    <div className="space-y-1">
      <p className="text-sm font-medium text-foreground">No image generated</p>
      <p className="text-xs text-muted-foreground">
        Enter a prompt below to begin creating
      </p>
    </div>
  </motion.div>
);

const LoadingState = () => (
  <motion.div
    initial={{ opacity: 0 }}
    animate={{ opacity: 1 }}
    exit={{ opacity: 0 }}
    className="flex flex-col items-center justify-center gap-6"
  >
    <div className="relative">
      {/* Outer glow ring */}
      <motion.div
        animate={{ scale: [1, 1.2, 1], opacity: [0.3, 0.1, 0.3] }}
        transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
        className="absolute inset-0 rounded-2xl bg-primary/20 blur-xl"
      />
      
      {/* Skeleton container */}
      <div className="relative h-40 w-40 sm:h-52 sm:w-52 md:h-64 md:w-64 overflow-hidden rounded-2xl border border-border bg-muted">
        <div className="absolute inset-0 animate-shimmer" />
        
        {/* Center icon */}
        <div className="absolute inset-0 flex items-center justify-center">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
          >
            <Sparkles className="h-8 w-8 text-primary animate-pulse-glow" />
          </motion.div>
        </div>
      </div>
    </div>
    
    <div className="space-y-2 text-center">
      <p className="text-sm font-medium text-foreground">Generating your image...</p>
      <div className="flex items-center justify-center gap-1.5">
        {[0, 1, 2].map((i) => (
          <motion.div
            key={i}
            animate={{ opacity: [0.3, 1, 0.3] }}
            transition={{ duration: 1, repeat: Infinity, delay: i * 0.2 }}
            className="h-1.5 w-1.5 rounded-full bg-primary"
          />
        ))}
      </div>
    </div>
  </motion.div>
);

const ImageResult = ({
  image,
  onDownload,
  onExpand,
}: {
  image: string;
  onDownload: () => void;
  onExpand: () => void;
}) => (
  <motion.div
    initial={{ opacity: 0, scale: 0.95 }}
    animate={{ opacity: 1, scale: 1 }}
    exit={{ opacity: 0, scale: 0.95 }}
    className="group relative"
  >
    <div className="relative overflow-hidden rounded-xl sm:rounded-2xl border border-border shadow-2xl shadow-black/50">
      <img
        src={image}
        alt="Generated artwork"
        className="block max-h-[40vh] sm:max-h-[50vh] md:max-h-[60vh] w-auto max-w-full object-contain"
      />
      
      {/* Hover overlay */}
      <motion.div
        initial={{ opacity: 0 }}
        whileHover={{ opacity: 1 }}
        className="absolute inset-0 flex items-end justify-center bg-gradient-to-t from-black/60 via-transparent to-transparent p-4"
      >
        <div className="flex gap-2">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={onDownload}
            className="flex h-8 sm:h-10 items-center gap-1.5 sm:gap-2 rounded-lg bg-foreground px-3 sm:px-4 text-xs sm:text-sm font-medium text-background transition-colors hover:bg-foreground/90"
          >
            <Download className="h-3.5 w-3.5 sm:h-4 sm:w-4" />
            <span className="hidden xs:inline">Download</span>
          </motion.button>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={onExpand}
            className="flex h-8 w-8 sm:h-10 sm:w-10 items-center justify-center rounded-lg border border-border/50 bg-background/80 backdrop-blur-sm transition-colors hover:bg-background"
          >
            <Expand className="h-3.5 w-3.5 sm:h-4 sm:w-4" />
          </motion.button>
        </div>
      </motion.div>
    </div>
  </motion.div>
);

export const Canvas = ({ image, isLoading, onDownload, onExpand }: CanvasProps) => {
  return (
    <div className="relative flex flex-1 items-center justify-center p-3 sm:p-5 md:p-8">
      {/* Viewport frame */}
      <div
        className={cn(
          "flex min-h-[250px] sm:min-h-[320px] md:min-h-[400px] w-full max-w-2xl items-center justify-center rounded-xl sm:rounded-2xl border border-dashed transition-colors duration-300",
          isLoading ? "border-primary/50" : "border-border"
        )}
      >
        <AnimatePresence mode="wait">
          {isLoading ? (
            <LoadingState key="loading" />
          ) : image ? (
            <ImageResult
              key="result"
              image={image}
              onDownload={onDownload}
              onExpand={onExpand}
            />
          ) : (
            <EmptyState key="empty" />
          )}
        </AnimatePresence>
      </div>

      {/* Loading progress bar */}
      <AnimatePresence>
        {isLoading && (
          <motion.div
            initial={{ scaleX: 0, opacity: 0 }}
            animate={{ scaleX: 1, opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute left-0 right-0 top-0 h-0.5 origin-left bg-gradient-to-r from-primary via-primary to-transparent"
            style={{
              background: "linear-gradient(90deg, hsl(var(--primary)), hsl(var(--primary) / 0.5), transparent)",
            }}
          />
        )}
      </AnimatePresence>
    </div>
  );
};
