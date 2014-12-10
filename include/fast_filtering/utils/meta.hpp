/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2014 Max-Planck-Institute for Intelligent Systems,
 *                     University of Southern California
 *    Jan Issac (jan.issac@gmail.com)
 *
 *
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

/**
 * @date 05/25/2014
 * @author Jan Issac (jan.issac@gmail.com)
 * Max-Planck-Institute for Intelligent Systems,
 *  University of Southern California
 */

#ifndef FAST_FILTERING_UTILS_META_HPP
#define FAST_FILTERING_UTILS_META_HPP

#include <Eigen/Dense>
#include <fast_filtering/utils/traits.hpp>

namespace fl
{

/**
 * \struct JoinSizes
 * \ingroup meta
 *
 * \tparam Sizes    Variadic argument containing a list of sizes. Each size
 *                  is a result of a constexpr or a const of a signed integral
 *                  type. Unknown sizes are represented by -1 or Eigen::Dynamic.
 *
 *
 * Joins the sizes within the \c Sizes argument list if all sizes are
 * non-dynamic. That is, all sizes are greater -1 (\c Eigen::Dynamic). If one of
 * the passed sizes is \c Eigen::Dynamic, the total size collapses to
 * \c Eigen::Dynamic.
 */
template <int... Sizes> struct JoinSizes;

/**
 * \ingroup meta
 * \sa JoinSizes
 * \copydoc JoinSizes
 *
 * \tparam Head     Head of the \c Sizes list of the previous recursive call
 *
 * Variadic counting recursion
 */
template <int Head, int... Sizes> struct JoinSizes<Head, Sizes...>
{
    enum
    {
        Size = IsFixed<Head>()
                   ? Head + JoinSizes<Sizes...>::Size
                   : Eigen::Dynamic
    };
};

/**
 * \internal
 *
 * \ingroup meta
 * \sa JoinSizes
 *
 * Recursion termination
 */
template <> struct JoinSizes<> { enum { Size = 0 }; } ;

}

#endif

