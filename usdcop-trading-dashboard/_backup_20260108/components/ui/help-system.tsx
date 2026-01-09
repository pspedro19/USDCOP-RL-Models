"use client";

import * as React from "react";
import * as Dialog from "@radix-ui/react-dialog";
import * as Tabs from "@radix-ui/react-tabs";
import { motion, AnimatePresence } from "framer-motion";
import {
  HelpCircle,
  Keyboard,
  BookOpen,
  Video,
  Search,
  ChevronRight,
  ChevronLeft,
  X,
  Play,
  Pause,
  RotateCcw,
  CheckCircle,
  AlertCircle,
  Info,
  ExternalLink,
  Download,
  Star,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";
import Fuse from "fuse.js";

interface HelpTopic {
  id: string;
  title: string;
  description: string;
  category: string;
  content: React.ReactNode;
  difficulty: "beginner" | "intermediate" | "advanced";
  estimatedTime: string;
  prerequisites?: string[];
  keywords: string[];
}

interface Tutorial {
  id: string;
  title: string;
  description: string;
  steps: TutorialStep[];
  category: string;
  difficulty: "beginner" | "intermediate" | "advanced";
  estimatedTime: string;
  thumbnail?: string;
}

interface TutorialStep {
  id: string;
  title: string;
  content: string;
  action?: string;
  target?: string;
  image?: string;
  video?: string;
  tip?: string;
}

interface ShortcutGroup {
  category: string;
  shortcuts: {
    key: string;
    description: string;
    example?: string;
  }[];
}

interface HelpSystemProps {
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  defaultTab?: string;
  className?: string;
}

const helpTopics: HelpTopic[] = [
  {
    id: "getting-started",
    title: "Getting Started",
    description: "Learn the basics of using the trading platform",
    category: "Basics",
    difficulty: "beginner",
    estimatedTime: "5 minutes",
    keywords: ["start", "begin", "intro", "basics"],
    content: (
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-white">Welcome to the Trading Platform</h3>
        <p className="text-gray-300">
          This platform provides professional-grade trading tools with Bloomberg Terminal-level functionality.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-4 bg-gray-800 rounded-lg">
            <h4 className="font-semibold text-white mb-2">Key Features</h4>
            <ul className="text-sm text-gray-300 space-y-1">
              <li>• Real-time price data</li>
              <li>• Advanced charting tools</li>
              <li>• Professional drawing tools</li>
              <li>• Risk management</li>
            </ul>
          </div>
          <div className="p-4 bg-gray-800 rounded-lg">
            <h4 className="font-semibold text-white mb-2">Quick Start</h4>
            <ul className="text-sm text-gray-300 space-y-1">
              <li>• Press Cmd+K for command palette</li>
              <li>• Right-click for context menu</li>
              <li>• Use keyboard shortcuts</li>
              <li>• Drag to pan, scroll to zoom</li>
            </ul>
          </div>
        </div>
      </div>
    ),
  },
  {
    id: "chart-navigation",
    title: "Chart Navigation",
    description: "Master chart navigation and interaction",
    category: "Charts",
    difficulty: "beginner",
    estimatedTime: "3 minutes",
    keywords: ["chart", "navigate", "zoom", "pan", "move"],
    content: (
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-white">Chart Navigation</h3>
        <div className="space-y-3">
          <div className="p-3 bg-gray-800 rounded">
            <strong className="text-white">Mouse Navigation:</strong>
            <ul className="text-sm text-gray-300 mt-2 space-y-1">
              <li>• Scroll wheel to zoom in/out</li>
              <li>• Drag to pan the chart</li>
              <li>• Right-click for context menu</li>
            </ul>
          </div>
          <div className="p-3 bg-gray-800 rounded">
            <strong className="text-white">Touch Navigation:</strong>
            <ul className="text-sm text-gray-300 mt-2 space-y-1">
              <li>• Pinch to zoom</li>
              <li>• Long press for context menu</li>
              <li>• Double tap to reset view</li>
            </ul>
          </div>
        </div>
      </div>
    ),
  },
  // Add more help topics...
];

const tutorials: Tutorial[] = [
  {
    id: "first-trade",
    title: "Making Your First Trade",
    description: "Step-by-step guide to placing your first order",
    category: "Trading",
    difficulty: "beginner",
    estimatedTime: "10 minutes",
    steps: [
      {
        id: "step-1",
        title: "Open Trading Panel",
        content: "Click on the trading panel or press 'B' for buy order",
        action: "Press B key or click Trading button",
        tip: "Use keyboard shortcuts for faster trading",
      },
      {
        id: "step-2",
        title: "Set Order Parameters",
        content: "Enter the amount, price, and order type",
        tip: "Always double-check your order before submitting",
      },
      {
        id: "step-3",
        title: "Review and Submit",
        content: "Review your order details and click submit",
        tip: "Consider setting stop-loss and take-profit levels",
      },
    ],
  },
  // Add more tutorials...
];

const shortcutGroups: ShortcutGroup[] = [
  {
    category: "General",
    shortcuts: [
      { key: "Cmd+K", description: "Open command palette", example: "Quick actions" },
      { key: "Cmd+/", description: "Show help", example: "Open this help system" },
      { key: "F11", description: "Toggle fullscreen", example: "Distraction-free mode" },
      { key: "Escape", description: "Close dialogs/cancel", example: "Exit modal or cancel action" },
    ],
  },
  {
    category: "Chart Navigation",
    shortcuts: [
      { key: "Space", description: "Pan mode", example: "Drag chart around" },
      { key: "+", description: "Zoom in", example: "Get closer view" },
      { key: "-", description: "Zoom out", example: "See more data" },
      { key: "0", description: "Reset zoom", example: "Back to default view" },
    ],
  },
  {
    category: "Timeframes",
    shortcuts: [
      { key: "1", description: "1 minute chart", example: "Scalping view" },
      { key: "5", description: "5 minute chart", example: "Short-term view" },
      { key: "H", description: "1 hour chart", example: "Medium-term view" },
      { key: "D", description: "Daily chart", example: "Long-term view" },
    ],
  },
  {
    category: "Trading",
    shortcuts: [
      { key: "B", description: "Buy order", example: "Open long position" },
      { key: "S", description: "Sell order", example: "Open short position" },
      { key: "X", description: "Close position", example: "Exit current trades" },
      { key: "Cmd+C", description: "Cancel orders", example: "Cancel pending orders" },
    ],
  },
  {
    category: "Drawing Tools",
    shortcuts: [
      { key: "T", description: "Trend line", example: "Draw trend lines" },
      { key: "F", description: "Fibonacci", example: "Fibonacci retracements" },
      { key: "R", description: "Rectangle", example: "Support/resistance zones" },
      { key: "L", description: "Line tool", example: "Draw horizontal/vertical lines" },
    ],
  },
];

const HelpSystem = React.forwardRef<
  React.ElementRef<typeof Dialog.Root>,
  HelpSystemProps
>(({ open, onOpenChange, defaultTab = "overview", className }, ref) => {
  const [activeTab, setActiveTab] = React.useState(defaultTab);
  const [searchQuery, setSearchQuery] = React.useState("");
  const [selectedTutorial, setSelectedTutorial] = React.useState<Tutorial | null>(null);
  const [currentStep, setCurrentStep] = React.useState(0);
  const [tutorialProgress, setTutorialProgress] = React.useState<Record<string, number>>({});

  // Keyboard shortcuts for help system
  const { shortcuts } = useKeyboardShortcuts({
    onSystemAction: (action) => {
      if (action === "help") {
        onOpenChange?.(!open);
      }
    },
  });

  // Fuse.js for fuzzy search
  const fuse = React.useMemo(() => {
    const searchData = [
      ...helpTopics.map(topic => ({ type: "topic", ...topic })),
      ...tutorials.map(tutorial => ({ type: "tutorial", ...tutorial })),
      ...shortcuts.map(shortcut => ({ type: "shortcut", ...shortcut })),
    ];

    return new Fuse(searchData, {
      keys: ["title", "description", "keywords", "key"],
      threshold: 0.3,
      includeScore: true,
    });
  }, [shortcuts]);

  // Filter content based on search
  const filteredContent = React.useMemo(() => {
    if (!searchQuery.trim()) {
      return {
        topics: helpTopics,
        tutorials,
        shortcuts: shortcutGroups,
      };
    }

    const results = fuse.search(searchQuery);
    const topics = results.filter(r => r.item.type === "topic").map(r => r.item as HelpTopic);
    const tutorialResults = results.filter(r => r.item.type === "tutorial").map(r => r.item as Tutorial);

    return {
      topics,
      tutorials: tutorialResults,
      shortcuts: shortcutGroups, // Keep all shortcuts for now
    };
  }, [searchQuery, fuse]);

  // Start tutorial
  const startTutorial = React.useCallback((tutorial: Tutorial) => {
    setSelectedTutorial(tutorial);
    setCurrentStep(0);
    setActiveTab("tutorials");
  }, []);

  // Next tutorial step
  const nextStep = React.useCallback(() => {
    if (selectedTutorial && currentStep < selectedTutorial.steps.length - 1) {
      setCurrentStep(prev => prev + 1);
    }
  }, [selectedTutorial, currentStep]);

  // Previous tutorial step
  const prevStep = React.useCallback(() => {
    if (currentStep > 0) {
      setCurrentStep(prev => prev - 1);
    }
  }, [currentStep]);

  // Complete tutorial
  const completeTutorial = React.useCallback(() => {
    if (selectedTutorial) {
      setTutorialProgress(prev => ({
        ...prev,
        [selectedTutorial.id]: selectedTutorial.steps.length,
      }));
      setSelectedTutorial(null);
      setCurrentStep(0);
    }
  }, [selectedTutorial]);

  // Get difficulty badge color
  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case "beginner": return "bg-green-500/20 text-green-300 border-green-500/30";
      case "intermediate": return "bg-yellow-500/20 text-yellow-300 border-yellow-500/30";
      case "advanced": return "bg-red-500/20 text-red-300 border-red-500/30";
      default: return "bg-gray-500/20 text-gray-300 border-gray-500/30";
    }
  };

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50" />
        <Dialog.Content
          ref={ref}
          className={cn(
            "fixed inset-4 md:inset-8 lg:inset-16 bg-gray-900 border border-gray-700 rounded-lg shadow-2xl z-50",
            "flex flex-col overflow-hidden",
            className
          )}
        >
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-gray-700">
            <div className="flex items-center space-x-3">
              <HelpCircle className="h-6 w-6 text-blue-400" />
              <div>
                <Dialog.Title className="text-xl font-semibold text-white">
                  Help & Support
                </Dialog.Title>
                <Dialog.Description className="text-sm text-gray-400">
                  Learn how to use the trading platform effectively
                </Dialog.Description>
              </div>
            </div>
            <Dialog.Close className="p-2 hover:bg-gray-800 rounded-lg transition-colors">
              <X className="h-5 w-5 text-gray-400" />
            </Dialog.Close>
          </div>

          {/* Search Bar */}
          <div className="p-4 border-b border-gray-700">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <input
                type="text"
                placeholder="Search help topics, tutorials, or shortcuts..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
              />
            </div>
          </div>

          {/* Content */}
          <div className="flex-1 overflow-hidden">
            <Tabs.Root
              value={activeTab}
              onValueChange={setActiveTab}
              className="h-full flex flex-col"
            >
              {/* Tab Navigation */}
              <Tabs.List className="flex border-b border-gray-700 px-6">
                <Tabs.Trigger
                  value="overview"
                  className="px-4 py-2 text-sm font-medium text-gray-400 hover:text-white data-[state=active]:text-white data-[state=active]:border-b-2 data-[state=active]:border-blue-400"
                >
                  <BookOpen className="h-4 w-4 mr-2" />
                  Overview
                </Tabs.Trigger>
                <Tabs.Trigger
                  value="tutorials"
                  className="px-4 py-2 text-sm font-medium text-gray-400 hover:text-white data-[state=active]:text-white data-[state=active]:border-b-2 data-[state=active]:border-blue-400"
                >
                  <Video className="h-4 w-4 mr-2" />
                  Tutorials
                </Tabs.Trigger>
                <Tabs.Trigger
                  value="shortcuts"
                  className="px-4 py-2 text-sm font-medium text-gray-400 hover:text-white data-[state=active]:text-white data-[state=active]:border-b-2 data-[state=active]:border-blue-400"
                >
                  <Keyboard className="h-4 w-4 mr-2" />
                  Shortcuts
                </Tabs.Trigger>
              </Tabs.List>

              {/* Tab Content */}
              <div className="flex-1 overflow-auto">
                <Tabs.Content value="overview" className="p-6">
                  <div className="space-y-6">
                    {/* Quick Start */}
                    <div className="bg-gradient-to-r from-blue-500/10 to-purple-500/10 p-6 rounded-lg border border-blue-500/20">
                      <h3 className="text-lg font-semibold text-white mb-3">Quick Start Guide</h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        <button
                          onClick={() => startTutorial(tutorials[0])}
                          className="p-4 bg-gray-800 rounded-lg hover:bg-gray-700 transition-colors text-left"
                        >
                          <Play className="h-5 w-5 text-green-400 mb-2" />
                          <div className="font-medium text-white">First Trade</div>
                          <div className="text-sm text-gray-400">Learn to place orders</div>
                        </button>
                        <button className="p-4 bg-gray-800 rounded-lg hover:bg-gray-700 transition-colors text-left">
                          <Keyboard className="h-5 w-5 text-blue-400 mb-2" />
                          <div className="font-medium text-white">Shortcuts</div>
                          <div className="text-sm text-gray-400">Master keyboard shortcuts</div>
                        </button>
                        <button className="p-4 bg-gray-800 rounded-lg hover:bg-gray-700 transition-colors text-left">
                          <BookOpen className="h-5 w-5 text-purple-400 mb-2" />
                          <div className="font-medium text-white">Help Topics</div>
                          <div className="text-sm text-gray-400">Browse all help articles</div>
                        </button>
                      </div>
                    </div>

                    {/* Help Topics */}
                    <div>
                      <h3 className="text-lg font-semibold text-white mb-4">Help Topics</h3>
                      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                        {filteredContent.topics.map((topic) => (
                          <div key={topic.id} className="bg-gray-800 rounded-lg p-4 hover:bg-gray-750 transition-colors">
                            <div className="flex items-start justify-between mb-2">
                              <h4 className="font-medium text-white">{topic.title}</h4>
                              <span className={cn(
                                "px-2 py-1 rounded text-xs border",
                                getDifficultyColor(topic.difficulty)
                              )}>
                                {topic.difficulty}
                              </span>
                            </div>
                            <p className="text-sm text-gray-400 mb-3">{topic.description}</p>
                            <div className="flex items-center justify-between text-xs text-gray-500">
                              <span>{topic.estimatedTime}</span>
                              <span>{topic.category}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </Tabs.Content>

                <Tabs.Content value="tutorials" className="p-6">
                  <AnimatePresence mode="wait">
                    {selectedTutorial ? (
                      <motion.div
                        key="tutorial-view"
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -20 }}
                        className="space-y-6"
                      >
                        {/* Tutorial Header */}
                        <div className="flex items-center space-x-4">
                          <button
                            onClick={() => setSelectedTutorial(null)}
                            className="p-2 hover:bg-gray-800 rounded-lg transition-colors"
                          >
                            <ChevronLeft className="h-5 w-5 text-gray-400" />
                          </button>
                          <div>
                            <h3 className="text-xl font-semibold text-white">{selectedTutorial.title}</h3>
                            <p className="text-sm text-gray-400">{selectedTutorial.description}</p>
                          </div>
                        </div>

                        {/* Progress Bar */}
                        <div className="bg-gray-800 rounded-lg p-4">
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-sm font-medium text-white">
                              Step {currentStep + 1} of {selectedTutorial.steps.length}
                            </span>
                            <span className="text-sm text-gray-400">
                              {Math.round(((currentStep + 1) / selectedTutorial.steps.length) * 100)}% complete
                            </span>
                          </div>
                          <div className="w-full bg-gray-700 rounded-full h-2">
                            <div
                              className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                              style={{ width: `${((currentStep + 1) / selectedTutorial.steps.length) * 100}%` }}
                            />
                          </div>
                        </div>

                        {/* Current Step */}
                        <div className="bg-gray-800 rounded-lg p-6">
                          <h4 className="text-lg font-semibold text-white mb-3">
                            {selectedTutorial.steps[currentStep].title}
                          </h4>
                          <p className="text-gray-300 mb-4">
                            {selectedTutorial.steps[currentStep].content}
                          </p>
                          {selectedTutorial.steps[currentStep].tip && (
                            <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-3 mb-4">
                              <div className="flex items-center space-x-2">
                                <Info className="h-4 w-4 text-blue-400" />
                                <span className="text-sm font-medium text-blue-300">Tip</span>
                              </div>
                              <p className="text-sm text-blue-200 mt-1">
                                {selectedTutorial.steps[currentStep].tip}
                              </p>
                            </div>
                          )}
                        </div>

                        {/* Navigation */}
                        <div className="flex items-center justify-between">
                          <button
                            onClick={prevStep}
                            disabled={currentStep === 0}
                            className="flex items-center space-x-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg transition-colors"
                          >
                            <ChevronLeft className="h-4 w-4" />
                            <span>Previous</span>
                          </button>

                          {currentStep === selectedTutorial.steps.length - 1 ? (
                            <button
                              onClick={completeTutorial}
                              className="flex items-center space-x-2 px-4 py-2 bg-green-500 hover:bg-green-600 text-white rounded-lg transition-colors"
                            >
                              <CheckCircle className="h-4 w-4" />
                              <span>Complete</span>
                            </button>
                          ) : (
                            <button
                              onClick={nextStep}
                              className="flex items-center space-x-2 px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors"
                            >
                              <span>Next</span>
                              <ChevronRight className="h-4 w-4" />
                            </button>
                          )}
                        </div>
                      </motion.div>
                    ) : (
                      <motion.div
                        key="tutorial-list"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="space-y-6"
                      >
                        <div>
                          <h3 className="text-lg font-semibold text-white mb-4">Interactive Tutorials</h3>
                          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                            {filteredContent.tutorials.map((tutorial) => (
                              <div key={tutorial.id} className="bg-gray-800 rounded-lg p-4">
                                <div className="flex items-start justify-between mb-3">
                                  <h4 className="font-medium text-white">{tutorial.title}</h4>
                                  <span className={cn(
                                    "px-2 py-1 rounded text-xs border",
                                    getDifficultyColor(tutorial.difficulty)
                                  )}>
                                    {tutorial.difficulty}
                                  </span>
                                </div>
                                <p className="text-sm text-gray-400 mb-4">{tutorial.description}</p>
                                <div className="flex items-center justify-between">
                                  <div className="flex items-center space-x-4 text-xs text-gray-500">
                                    <span>{tutorial.estimatedTime}</span>
                                    <span>{tutorial.steps.length} steps</span>
                                  </div>
                                  <button
                                    onClick={() => startTutorial(tutorial)}
                                    className="flex items-center space-x-1 px-3 py-1 bg-blue-500 hover:bg-blue-600 text-white rounded text-sm transition-colors"
                                  >
                                    <Play className="h-3 w-3" />
                                    <span>Start</span>
                                  </button>
                                </div>
                                {tutorialProgress[tutorial.id] > 0 && (
                                  <div className="mt-3 w-full bg-gray-700 rounded-full h-1">
                                    <div
                                      className="bg-green-500 h-1 rounded-full"
                                      style={{ width: `${(tutorialProgress[tutorial.id] / tutorial.steps.length) * 100}%` }}
                                    />
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </Tabs.Content>

                <Tabs.Content value="shortcuts" className="p-6">
                  <div className="space-y-6">
                    <div>
                      <h3 className="text-lg font-semibold text-white mb-4">Keyboard Shortcuts</h3>
                      <p className="text-gray-400 mb-6">Master these shortcuts to trade like a pro</p>
                    </div>

                    {filteredContent.shortcuts.map((group) => (
                      <div key={group.category} className="bg-gray-800 rounded-lg p-4">
                        <h4 className="font-semibold text-white mb-4">{group.category}</h4>
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                          {group.shortcuts.map((shortcut, index) => (
                            <div key={index} className="flex items-center justify-between py-2">
                              <div>
                                <div className="text-white">{shortcut.description}</div>
                                {shortcut.example && (
                                  <div className="text-xs text-gray-400">{shortcut.example}</div>
                                )}
                              </div>
                              <kbd className="px-2 py-1 bg-gray-700 border border-gray-600 rounded text-sm font-mono text-gray-300">
                                {shortcut.key}
                              </kbd>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </Tabs.Content>
              </div>
            </Tabs.Root>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
});

HelpSystem.displayName = "HelpSystem";

export { HelpSystem };