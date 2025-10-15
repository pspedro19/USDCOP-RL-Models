// Basic animation variants
export const fadeIn = {
  initial: { opacity: 0 },
  animate: { opacity: 1 },
  exit: { opacity: 0 }
}

export const slideIn = {
  initial: { x: -20, opacity: 0 },
  animate: { x: 0, opacity: 1 },
  exit: { x: 20, opacity: 0 }
}

// Professional easing presets for Bloomberg Terminal-style UI
export const easing = {
  smooth: [0.4, 0, 0.2, 1], // Standard smooth transitions
  bounce: [0.68, -0.55, 0.265, 1.55], // Playful bounce effect
  snappy: [0.25, 0.46, 0.45, 0.94], // Quick, responsive transitions
  linear: [0, 0, 1, 1], // Linear progression
  terminal: [0.4, 0, 0.2, 1], // Terminal-optimized easing
  professional: [0.25, 0.1, 0.25, 1], // Professional Bloomberg-style
  glass: [0.16, 1, 0.3, 1], // Glassmorphism optimized
  elastic: [0.175, 0.885, 0.32, 1.275], // Elastic feedback
  expo: [0.19, 1, 0.22, 1], // Exponential ease
  back: [0.68, -0.6, 0.32, 1.6] // Back ease with overshoot
}

// Component-specific animations
export const components = {
  glassButton: {
    initial: {
      scale: 1,
      boxShadow: '0 4px 16px rgba(0, 0, 0, 0.1)'
    },
    hover: {
      scale: 1.02,
      boxShadow: '0 8px 32px rgba(6, 182, 212, 0.15)',
      transition: { duration: 0.2, ease: easing.glass }
    },
    tap: {
      scale: 0.98,
      transition: { duration: 0.1, ease: easing.expo }
    }
  },

  hoverCard: {
    initial: {
      scale: 1,
      y: 0,
      boxShadow: '0 4px 16px rgba(0, 0, 0, 0.1)'
    },
    hover: {
      scale: 1.01,
      y: -2,
      boxShadow: '0 8px 32px rgba(6, 182, 212, 0.15)',
      transition: { duration: 0.3, ease: easing.professional }
    },
    tap: {
      scale: 0.99,
      transition: { duration: 0.1, ease: easing.expo }
    }
  },

  // Bloomberg Terminal Professional Components
  terminalButton: {
    initial: {
      scale: 1,
      background: 'linear-gradient(135deg, rgba(6, 182, 212, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%)',
      boxShadow: '0 4px 16px rgba(0, 0, 0, 0.1)',
      border: '1px solid rgba(6, 182, 212, 0.2)'
    },
    hover: {
      scale: 1.02,
      background: 'linear-gradient(135deg, rgba(6, 182, 212, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%)',
      boxShadow: '0 8px 32px rgba(6, 182, 212, 0.25)',
      border: '1px solid rgba(6, 182, 212, 0.4)',
      transition: { duration: 0.3, ease: easing.professional }
    },
    tap: {
      scale: 0.98,
      transition: { duration: 0.1, ease: easing.expo }
    },
    disabled: {
      opacity: 0.5,
      scale: 1,
      transition: { duration: 0.2, ease: easing.smooth }
    }
  },

  priceUpdateCard: {
    initial: {
      scale: 1,
      backgroundColor: 'rgba(15, 20, 27, 0.85)',
      borderColor: 'rgba(100, 116, 139, 0.3)'
    },
    priceUp: {
      scale: [1, 1.02, 1],
      backgroundColor: [
        'rgba(15, 20, 27, 0.85)',
        'rgba(0, 211, 149, 0.2)',
        'rgba(15, 20, 27, 0.85)'
      ],
      borderColor: [
        'rgba(100, 116, 139, 0.3)',
        'rgba(0, 211, 149, 0.6)',
        'rgba(100, 116, 139, 0.3)'
      ],
      transition: { duration: 0.8, ease: easing.professional }
    },
    priceDown: {
      scale: [1, 1.02, 1],
      backgroundColor: [
        'rgba(15, 20, 27, 0.85)',
        'rgba(255, 59, 105, 0.2)',
        'rgba(15, 20, 27, 0.85)'
      ],
      borderColor: [
        'rgba(100, 116, 139, 0.3)',
        'rgba(255, 59, 105, 0.6)',
        'rgba(100, 116, 139, 0.3)'
      ],
      transition: { duration: 0.8, ease: easing.professional }
    }
  },

  metricCard: {
    initial: {
      opacity: 0,
      scale: 0.95,
      y: 20
    },
    animate: {
      opacity: 1,
      scale: 1,
      y: 0,
      transition: {
        duration: 0.5,
        ease: easing.professional
      }
    },
    hover: {
      scale: 1.02,
      y: -4,
      transition: { duration: 0.2, ease: easing.glass }
    },
    updated: {
      scale: [1, 1.05, 1],
      boxShadow: [
        '0 4px 16px rgba(0, 0, 0, 0.1)',
        '0 8px 32px rgba(6, 182, 212, 0.3)',
        '0 4px 16px rgba(0, 0, 0, 0.1)'
      ],
      transition: { duration: 0.8, ease: easing.professional }
    }
  },

  tradingIndicator: {
    initial: { scale: 1, opacity: 0.8 },
    live: {
      scale: [1, 1.1, 1],
      opacity: [0.8, 1, 0.8],
      transition: {
        duration: 2,
        repeat: Infinity,
        ease: easing.smooth
      }
    },
    error: {
      scale: [1, 1.2, 1],
      opacity: [0.8, 1, 0.8],
      transition: {
        duration: 1,
        repeat: Infinity,
        ease: easing.bounce
      }
    }
  },

  glassMorphCard: {
    initial: {
      opacity: 0,
      scale: 0.9,
      y: 30,
      backdropFilter: 'blur(0px)'
    },
    animate: {
      opacity: 1,
      scale: 1,
      y: 0,
      backdropFilter: 'blur(16px)',
      transition: {
        duration: 0.6,
        ease: easing.glass
      }
    },
    hover: {
      scale: 1.01,
      y: -2,
      backdropFilter: 'blur(24px)',
      transition: { duration: 0.3, ease: easing.glass }
    },
    exit: {
      opacity: 0,
      scale: 0.95,
      y: 20,
      backdropFilter: 'blur(0px)',
      transition: { duration: 0.4, ease: easing.expo }
    }
  }
}

// Loading animations
export const loading = {
  spinner: {
    initial: { rotate: 0 },
    animate: {
      rotate: 360,
      transition: {
        duration: 1,
        repeat: Infinity,
        ease: 'linear'
      }
    }
  },

  terminalSpinner: {
    initial: {
      rotate: 0,
      scale: 1,
      boxShadow: '0 0 0px rgba(6, 182, 212, 0)'
    },
    animate: {
      rotate: 360,
      scale: [1, 1.05, 1],
      boxShadow: [
        '0 0 0px rgba(6, 182, 212, 0)',
        '0 0 16px rgba(6, 182, 212, 0.6)',
        '0 0 0px rgba(6, 182, 212, 0)'
      ],
      transition: {
        rotate: {
          duration: 2,
          repeat: Infinity,
          ease: 'linear'
        },
        scale: {
          duration: 2,
          repeat: Infinity,
          ease: easing.smooth
        },
        boxShadow: {
          duration: 2,
          repeat: Infinity,
          ease: easing.smooth
        }
      }
    }
  },

  shimmer: {
    initial: { x: '-100%' },
    animate: {
      x: '100%',
      transition: {
        duration: 1.5,
        repeat: Infinity,
        ease: 'linear'
      }
    }
  },

  glassShimmer: {
    initial: {
      x: '-100%',
      opacity: 0
    },
    animate: {
      x: '100%',
      opacity: [0, 0.6, 0],
      transition: {
        duration: 2,
        repeat: Infinity,
        ease: 'linear'
      }
    }
  },

  pulse: {
    initial: { opacity: 0.6 },
    animate: {
      opacity: 1,
      transition: {
        duration: 1.2,
        repeat: Infinity,
        repeatType: 'reverse',
        ease: easing.smooth
      }
    }
  },

  terminalPulse: {
    initial: {
      opacity: 0.6,
      textShadow: '0 0 0px rgba(6, 182, 212, 0)'
    },
    animate: {
      opacity: [0.6, 1, 0.6],
      textShadow: [
        '0 0 0px rgba(6, 182, 212, 0)',
        '0 0 12px rgba(6, 182, 212, 0.8)',
        '0 0 0px rgba(6, 182, 212, 0)'
      ],
      transition: {
        duration: 2,
        repeat: Infinity,
        ease: easing.professional
      }
    }
  },

  dataStream: {
    initial: {
      opacity: 0,
      x: -20,
      scale: 0.9
    },
    animate: {
      opacity: [0, 1, 1, 0],
      x: [-20, 0, 0, 20],
      scale: [0.9, 1, 1, 0.9],
      transition: {
        duration: 3,
        repeat: Infinity,
        ease: easing.professional
      }
    }
  },

  skeleton: {
    initial: {
      background: 'linear-gradient(90deg, rgba(51, 65, 85, 0.1) 25%, rgba(6, 182, 212, 0.1) 50%, rgba(51, 65, 85, 0.1) 75%)'
    },
    animate: {
      backgroundPosition: ['0% 50%', '100% 50%', '0% 50%'],
      transition: {
        duration: 2,
        repeat: Infinity,
        ease: 'linear'
      }
    }
  }
}

// List animations
export const lists = {
  container: {
    initial: {},
    animate: {
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.2
      }
    }
  },

  item: {
    initial: {
      opacity: 0,
      y: 20
    },
    animate: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.4,
        ease: easing.smooth
      }
    }
  }
}

// Utility functions
export const utils = {
  createSlideAnimation: (direction: 'up' | 'down' | 'left' | 'right', distance: number = 20) => {
    const directionMap = {
      up: { y: distance },
      down: { y: -distance },
      left: { x: distance },
      right: { x: -distance }
    }

    return {
      initial: {
        opacity: 0,
        ...directionMap[direction]
      },
      animate: {
        opacity: 1,
        x: 0,
        y: 0,
        transition: {
          duration: 0.4,
          ease: easing.professional
        }
      },
      exit: {
        opacity: 0,
        ...directionMap[direction],
        transition: {
          duration: 0.3,
          ease: easing.expo
        }
      }
    }
  },

  createScaleAnimation: (scale: number = 0.95) => ({
    initial: {
      opacity: 0,
      scale
    },
    animate: {
      opacity: 1,
      scale: 1,
      transition: {
        duration: 0.3,
        ease: easing.professional
      }
    },
    exit: {
      opacity: 0,
      scale: scale * 0.9,
      transition: {
        duration: 0.2,
        ease: easing.expo
      }
    }
  }),

  createGlassAnimation: (direction: 'up' | 'down' | 'left' | 'right' = 'up') => {
    const directionMap = {
      up: { y: 30 },
      down: { y: -30 },
      left: { x: 30 },
      right: { x: -30 }
    }

    return {
      initial: {
        opacity: 0,
        scale: 0.9,
        backdropFilter: 'blur(0px)',
        ...directionMap[direction]
      },
      animate: {
        opacity: 1,
        scale: 1,
        x: 0,
        y: 0,
        backdropFilter: 'blur(16px)',
        transition: {
          duration: 0.6,
          ease: easing.glass
        }
      },
      exit: {
        opacity: 0,
        scale: 0.95,
        backdropFilter: 'blur(0px)',
        ...directionMap[direction],
        transition: {
          duration: 0.4,
          ease: easing.expo
        }
      }
    }
  },

  createTerminalAnimation: (delay: number = 0) => ({
    initial: {
      opacity: 0,
      y: 10,
      textShadow: '0 0 0px rgba(6, 182, 212, 0)'
    },
    animate: {
      opacity: 1,
      y: 0,
      textShadow: '0 0 8px rgba(6, 182, 212, 0.3)',
      transition: {
        duration: 0.5,
        delay,
        ease: easing.terminal
      }
    }
  }),

  createPriceFlashAnimation: (direction: 'up' | 'down') => ({
    initial: {
      scale: 1,
      backgroundColor: 'rgba(15, 20, 27, 0.85)'
    },
    flash: {
      scale: [1, 1.02, 1],
      backgroundColor: direction === 'up'
        ? ['rgba(15, 20, 27, 0.85)', 'rgba(0, 211, 149, 0.3)', 'rgba(15, 20, 27, 0.85)']
        : ['rgba(15, 20, 27, 0.85)', 'rgba(255, 59, 105, 0.3)', 'rgba(15, 20, 27, 0.85)'],
      boxShadow: direction === 'up'
        ? ['0 0 0px rgba(0, 211, 149, 0)', '0 0 20px rgba(0, 211, 149, 0.6)', '0 0 8px rgba(0, 211, 149, 0.3)']
        : ['0 0 0px rgba(255, 59, 105, 0)', '0 0 20px rgba(255, 59, 105, 0.6)', '0 0 8px rgba(255, 59, 105, 0.3)'],
      transition: {
        duration: 0.8,
        ease: easing.professional
      }
    }
  }),

  createStaggerContainer: (staggerDelay: number = 0.1) => ({
    initial: {},
    animate: {
      transition: {
        staggerChildren: staggerDelay,
        delayChildren: 0.1
      }
    },
    exit: {
      transition: {
        staggerChildren: staggerDelay * 0.5,
        staggerDirection: -1
      }
    }
  }),

  createMicroInteraction: (scale: number = 1.02) => ({
    whileHover: {
      scale,
      transition: { duration: 0.2, ease: easing.glass }
    },
    whileTap: {
      scale: scale * 0.98,
      transition: { duration: 0.1, ease: easing.expo }
    }
  })
}

// Trading-specific animation presets
export const trading = {
  priceUp: utils.createPriceFlashAnimation('up'),
  priceDown: utils.createPriceFlashAnimation('down'),
  marketOpen: {
    initial: { scale: 0.9, opacity: 0 },
    animate: {
      scale: 1,
      opacity: 1,
      transition: { duration: 0.5, ease: easing.professional }
    }
  },
  marketClose: {
    initial: { scale: 1, opacity: 1 },
    animate: {
      scale: 0.95,
      opacity: 0.6,
      transition: { duration: 0.3, ease: easing.expo }
    }
  },
  orderFilled: {
    scale: [1, 1.1, 1],
    boxShadow: [
      '0 0 0px rgba(0, 211, 149, 0)',
      '0 0 20px rgba(0, 211, 149, 0.8)',
      '0 0 8px rgba(0, 211, 149, 0.3)'
    ],
    transition: { duration: 0.6, ease: easing.bounce }
  },
  orderCancelled: {
    scale: [1, 0.95, 1],
    opacity: [1, 0.5, 1],
    transition: { duration: 0.4, ease: easing.professional }
  }
}

// Professional Bloomberg Terminal effects
export const terminal = {
  boot: {
    initial: {
      opacity: 0,
      scale: 0.98,
      filter: 'brightness(0)'
    },
    animate: {
      opacity: 1,
      scale: 1,
      filter: 'brightness(1)',
      transition: {
        duration: 1.2,
        ease: easing.professional
      }
    }
  },

  dataFeed: {
    initial: { opacity: 0, x: -10 },
    animate: {
      opacity: [0, 1, 1, 0],
      x: [-10, 0, 0, 10],
      transition: {
        duration: 2,
        repeat: Infinity,
        ease: easing.terminal
      }
    }
  },

  alertCritical: {
    scale: [1, 1.05, 1],
    borderColor: [
      'rgba(239, 68, 68, 0.3)',
      'rgba(239, 68, 68, 0.8)',
      'rgba(239, 68, 68, 0.3)'
    ],
    transition: {
      duration: 0.6,
      repeat: 3,
      ease: easing.bounce
    }
  },

  statusLive: {
    scale: [1, 1.1, 1],
    opacity: [0.8, 1, 0.8],
    boxShadow: [
      '0 0 0px rgba(16, 185, 129, 0)',
      '0 0 16px rgba(16, 185, 129, 0.6)',
      '0 0 8px rgba(16, 185, 129, 0.3)'
    ],
    transition: {
      duration: 2,
      repeat: Infinity,
      ease: easing.smooth
    }
  }
}

// Presets collection
export const presets = {
  easing,
  fadeIn,
  slideIn,
  scaleIn: utils.createScaleAnimation(),
  slideUp: utils.createSlideAnimation('up'),
  slideDown: utils.createSlideAnimation('down'),
  slideLeft: utils.createSlideAnimation('left'),
  slideRight: utils.createSlideAnimation('right'),
  glassUp: utils.createGlassAnimation('up'),
  glassDown: utils.createGlassAnimation('down'),
  glassLeft: utils.createGlassAnimation('left'),
  glassRight: utils.createGlassAnimation('right'),
  terminalText: utils.createTerminalAnimation(),
  microHover: utils.createMicroInteraction(),
  staggerFast: utils.createStaggerContainer(0.05),
  staggerNormal: utils.createStaggerContainer(0.1),
  staggerSlow: utils.createStaggerContainer(0.2)
}

// Complete motion library export
export const motionLibrary = {
  components,
  loading,
  lists,
  utils,
  presets,
  trading,
  terminal
}
